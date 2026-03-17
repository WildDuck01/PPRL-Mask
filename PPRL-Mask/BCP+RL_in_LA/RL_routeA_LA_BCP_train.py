from asyncore import write
import imp
import os
from sre_parse import SPECIAL_CHARS
import sys
from xml.etree.ElementInclude import default_loader
from tqdm import tqdm
from tensorboardX import SummaryWriter
import shutil
import argparse
import logging
import random
import numpy as np
from medpy import metric
import torch
import torch.optim as optim
from torchvision import transforms
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.nn as nn
import pdb

from yaml import parse
from skimage.measure import label
from torch.utils.data import DataLoader
from torch.autograd import Variable
from utils import losses, ramps, feature_memory, contrastive_losses, test_3d_patch
from dataloaders.dataset import *
from networks.net_factory import net_factory
from utils.BCP_utils import context_mask, mix_loss, parameter_sharing, update_ema_variables

from bcp_rl_routeA_utils import *
from rl_agent_bandit_routeA import BanditConfig, ContextualBanditAgent, extract_state_from_encoder_feature
from collections import defaultdict

import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# PyTorch 2.x 才有；旧版本直接跳过
if hasattr(torch, "set_float32_matmul_precision"):
    torch.set_float32_matmul_precision("high")



parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='/data/byh_data/SSNet_data/LA', help='Name of Dataset')
parser.add_argument('--exp', type=str,  default='BCP', help='exp_name')
parser.add_argument('--model', type=str, default='VNet', help='model_name')
parser.add_argument('--pre_max_iteration', type=int,  default=2000, help='maximum pre-train iteration to train')
parser.add_argument('--self_max_iteration', type=int,  default=15000, help='maximum self-train iteration to train')
parser.add_argument('--max_samples', type=int,  default=80, help='maximum samples to train')
parser.add_argument('--labeled_bs', type=int, default=4, help='batch_size of labeled data per gpu')
parser.add_argument('--batch_size', type=int, default=8, help='batch_size per gpu')
parser.add_argument('--base_lr', type=float,  default=0.01, help='maximum epoch number to train')
parser.add_argument('--deterministic', type=int,  default=1, help='whether use deterministic training')
parser.add_argument('--labelnum', type=int,  default=8, help='trained samples')
parser.add_argument('--gpu', type=str,  default='1', help='GPU to use')
parser.add_argument('--seed', type=int,  default=1337, help='random seed')
parser.add_argument('--consistency', type=float, default=1.0, help='consistency')
parser.add_argument('--consistency_rampup', type=float, default=40.0, help='consistency_rampup')
parser.add_argument('--magnitude', type=float,  default='10.0', help='magnitude')
# -- setting of BCP
parser.add_argument('--u_weight', type=float, default=0.5, help='weight of unlabeled pixels')
parser.add_argument('--mask_ratio', type=float, default=2/3, help='ratio of mask/image')
# -- setting of mixup
parser.add_argument('--u_alpha', type=float, default=2.0, help='unlabeled image ratio of mixuped image')
parser.add_argument('--loss_weight', type=float, default=0.5, help='loss weight of unimage term')


parser.add_argument('--rl_warmup_iters', type=int, default=1500, help='random actions before RL starts')

args = parser.parse_args()


lambda_edge: float = 0.05
ema_momentum_reward: float = 0.99
reward_clip: float = 0.2
reward_scale: float = 5.0


def extract_state_from_encoder_feature_3d(
    feature_map: torch.Tensor,
    entropy_mean: float,
    disagreement_mean: float,
) -> np.ndarray:
    """
    feature_map: [B,C,d,h,w] (from VNet forward second output)
    return: 1D numpy state [C+2]
    """
    with torch.no_grad():
        pooled = F.adaptive_avg_pool3d(feature_map, 1).squeeze(-1).squeeze(-1).squeeze(-1)  # [B,C]
        pooled = pooled.mean(dim=0, keepdim=True)  # [1,C]

        metrics = torch.tensor([[entropy_mean, disagreement_mean]], dtype=pooled.dtype, device=pooled.device)
        state = torch.cat([pooled, metrics], dim=1)  # [1,C+2]
        return state.squeeze(0).detach().cpu().numpy()

def get_cut_mask(out, thres=0.5, nms=0):
    probs = F.softmax(out, 1)
    masks = (probs >= thres).type(torch.int64)
    masks = masks[:, 1, :, :].contiguous()
    if nms == 1:
        masks = LargestCC_pancreas(masks)
    return masks

def LargestCC_pancreas(segmentation):
    N = segmentation.shape[0]
    batch_list = []
    for n in range(N):
        n_prob = segmentation[n].detach().cpu().numpy()
        labels = label(n_prob)
        if labels.max() != 0:
            largestCC = labels == np.argmax(np.bincount(labels.flat)[1:])+1
        else:
            largestCC = n_prob
        batch_list.append(largestCC)
    
    return torch.Tensor(batch_list).cuda()

def save_net_opt(net, optimizer, path):
    state = {
        'net': net.state_dict(),
        'opt': optimizer.state_dict(),
    }
    torch.save(state, str(path))

def load_net_opt(net, optimizer, path):
    state = torch.load(str(path))
    net.load_state_dict(state['net'])
    optimizer.load_state_dict(state['opt'])

def load_net(net, path):
    state = torch.load(str(path))
    net.load_state_dict(state['net'])

def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)

train_data_path = args.root_path

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
pre_max_iterations = args.pre_max_iteration
self_max_iterations = args.self_max_iteration
base_lr = args.base_lr
CE = nn.CrossEntropyLoss(reduction='none')

if args.deterministic:
    cudnn.benchmark = False
    cudnn.deterministic = True
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

patch_size = (112, 112, 80)
num_classes = 2

def pre_train(args, snapshot_path):
    model = net_factory(net_type=args.model, in_chns=1, class_num=num_classes, mode="train")
    db_train = LAHeart(base_dir=train_data_path,
                       split='train',
                       transform = transforms.Compose([
                          RandomRotFlip(),
                          RandomCrop(patch_size),
                          ToTensor(),
                          ]))
    labelnum = args.labelnum
    labeled_idxs = list(range(labelnum))
    unlabeled_idxs = list(range(labelnum, args.max_samples))
    batch_sampler = TwoStreamBatchSampler(labeled_idxs, unlabeled_idxs, args.batch_size, args.batch_size-args.labeled_bs)
    sub_bs = int(args.labeled_bs/2)
    def worker_init_fn(worker_id):
        random.seed(args.seed+worker_id)
    trainloader = DataLoader(
        db_train,
        batch_sampler=batch_sampler,
        num_workers=16,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4,
        worker_init_fn=worker_init_fn
        )
    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    DICE = losses.mask_DiceLoss(nclass=2)

    model.train()
    writer = SummaryWriter(snapshot_path+'/log')
    logging.info("{} iterations per epoch".format(len(trainloader)))
    iter_num = 0
    best_dice = 0
    max_epoch = pre_max_iterations // len(trainloader) + 1
    iterator = tqdm(range(max_epoch), ncols=70)
    for epoch_num in iterator:
        for _, sampled_batch in enumerate(trainloader):
            volume_batch, label_batch = sampled_batch['image'][:args.labeled_bs], sampled_batch['label'][:args.labeled_bs]
            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()
            img_a, img_b = volume_batch[:sub_bs], volume_batch[sub_bs:]
            lab_a, lab_b = label_batch[:sub_bs], label_batch[sub_bs:]
            with torch.no_grad():
                img_mask, loss_mask = context_mask(img_a, args.mask_ratio)

            """Mix Input"""
            volume_batch = img_a * img_mask + img_b * (1 - img_mask)
            label_batch = lab_a * img_mask + lab_b * (1 - img_mask)

            outputs, _ = model(volume_batch)
            loss_ce = F.cross_entropy(outputs, label_batch)
            loss_dice = DICE(outputs, label_batch)
            loss = (loss_ce + loss_dice) / 2

            iter_num += 1
            writer.add_scalar('pre/loss_dice', loss_dice, iter_num)
            writer.add_scalar('pre/loss_ce', loss_ce, iter_num)
            writer.add_scalar('pre/loss_all', loss, iter_num)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            logging.info('iteration %d : loss: %03f, loss_dice: %03f, loss_ce: %03f'%(iter_num, loss, loss_dice, loss_ce))

            if iter_num % 200 == 0:
                model.eval()
                dice_sample = test_3d_patch.var_all_case_LA(model, num_classes=num_classes, patch_size=patch_size, stride_xy=18, stride_z=4)
                if dice_sample > best_dice:
                    best_dice = round(dice_sample, 4)
                    save_mode_path = os.path.join(snapshot_path,  'iter_{}_dice_{}.pth'.format(iter_num, best_dice))
                    save_best_path = os.path.join(snapshot_path,'{}_best_model.pth'.format(args.model))
                    save_net_opt(model, optimizer, save_mode_path)
                    save_net_opt(model, optimizer, save_best_path)
                    # torch.save(model.state_dict(), save_mode_path)
                    # torch.save(model.state_dict(), save_best_path)
                    logging.info("save best model to {}".format(save_mode_path))
                writer.add_scalar('4_Var_dice/Dice', dice_sample, iter_num)
                writer.add_scalar('4_Var_dice/Best_dice', best_dice, iter_num)
                model.train()

            if iter_num >= pre_max_iterations:
                break

        if iter_num >= pre_max_iterations:
            iterator.close()
            break
    writer.close()


def self_train(args, pre_snapshot_path, self_snapshot_path):
    from torch.cuda.amp import autocast, GradScaler
    scaler = GradScaler()

    model = net_factory(net_type=args.model, in_chns=1, class_num=num_classes, mode="train",
                        returnEncoderFeature=True)
    ema_model = net_factory(net_type=args.model, in_chns=1, class_num=num_classes, mode="train")
    for param in ema_model.parameters():
            param.detach_()   # ema_model set
    db_train = LAHeart(base_dir=train_data_path,
                       split='train',
                       transform = transforms.Compose([
                          RandomRotFlip(),
                          RandomCrop(patch_size),
                          ToTensor(),
                          ]))
    labelnum = args.labelnum
    labeled_idxs = list(range(labelnum))
    unlabeled_idxs = list(range(labelnum, args.max_samples))
    batch_sampler = TwoStreamBatchSampler(labeled_idxs, unlabeled_idxs, args.batch_size, args.batch_size-args.labeled_bs)
    sub_bs = int(args.labeled_bs/2)
    def worker_init_fn(worker_id):
        random.seed(args.seed+worker_id)
    trainloader = DataLoader(db_train, batch_sampler=batch_sampler, num_workers=4, pin_memory=True, worker_init_fn=worker_init_fn)
    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)

    pretrained_model = os.path.join(pre_snapshot_path, f'{args.model}_best_model.pth')
    load_net(model, pretrained_model)
    load_net(ema_model, pretrained_model)
    
    model.train()
    ema_model.train()
    writer = SummaryWriter(self_snapshot_path+'/log')
    logging.info("{} iterations per epoch".format(len(trainloader)))
    iter_num = 0
    best_dice = 0
    max_epoch = self_max_iterations // len(trainloader) + 1
    lr_ = base_lr
    iterator = tqdm(range(max_epoch), ncols=70)
    
    # --- RL Module ---
    beta: float = args.mask_ratio  # 与原版 BCP 的 mask_ratio 对齐，避免改变 copy-paste 强度
    bandit_cfg = BanditConfig(state_dim=258, action_dim=4, lr=1e-4)
    rl_agent = ContextualBanditAgent(bandit_cfg, seed=args.seed + 1)
    
    # 关键：guided_beta 必须与 beta 一致，避免 RL 偏好“小 mask”走捷径
    mask_cfg = MaskActionConfig(
        beta=beta,
        guided_beta=beta,
        topk_percent=0.02,   # 你若原来不是 0.02，就保留原值
        per_sample=True
    )
    
    # --- reward EMA baseline ---
    ema_score:float = 0.0
    
    action_counts = defaultdict(int)
    action_reward_sum = defaultdict(float)
    
    
    for epoch in iterator:
        for _, sampled_batch in enumerate(trainloader):
            volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            volume_batch = volume_batch.cuda(non_blocking=True)
            label_batch  = label_batch.cuda(non_blocking=True)

            img_a, img_b = volume_batch[:sub_bs], volume_batch[sub_bs:args.labeled_bs]
            lab_a, lab_b = label_batch[:sub_bs], label_batch[sub_bs:args.labeled_bs]
            unimg_a, unimg_b = volume_batch[args.labeled_bs:args.labeled_bs+sub_bs], volume_batch[args.labeled_bs+sub_bs:]
            
            
            # ================================
            # (1) Teacher pseudo labels (BCP) 打上伪标签
            # ================================
            with torch.no_grad():
                teacher_logits_a, _ = ema_model(unimg_a)
                teacher_logits_b, _ = ema_model(unimg_b)
                plab_a = get_cut_mask(teacher_logits_a, nms=1)
                plab_b = get_cut_mask(teacher_logits_b, nms=1)
                
                
            # ================================
            # (2) Build RL state + choose action
            #     State strictly uses available signals (teacher+student),
            #     does NOT affect BCP training semantics.
            # ================================
            model_was_training = model.training # 记录模型在执行某段代码之前的模式，以便之后可以恢复原状态。
            model.eval()
            with torch.no_grad():
                with autocast(enabled=False):
                    # Student logits on the SAME image used for maps (uimg_a)
                    # 熵高：分布平坦（模型不确定）
                    # 熵低：分布尖锐（模型自信）
                    student_logits_a, _, enc_feat_a = model(unimg_a)  # 学生模型输出的 logits（未经过 softmax 的原始分类分数）
                    pT = F.softmax(teacher_logits_a, dim=1) # 得到类别概率分布 
                    pS = F.softmax(student_logits_a, dim=1)
                    # 计算 教师模型的熵 -- teacher输出的熵代表“伪标签可靠度”
                    ent_mean = float((-(pT * torch.log(torch.clamp(pT, 1e-6, 1.0))).sum(dim=1)).mean().item())
                    # 用二者熵计算 --- KL 散度 --- teacher/student二者的意见分歧程度
                    dis_mean = float((pT * (torch.log(torch.clamp(pT, 1e-6, 1.0)) - torch.log(torch.clamp(pS, 1e-6, 1.0)))).sum(dim=1).mean().item())
                    state = extract_state_from_encoder_feature_3d(enc_feat_a, ent_mean, dis_mean)
            
            if model_was_training: # 之前是在train状态被保存下来的 所以接下来会被转换成train模式
                model.train()
                
            # Warm-up: use purely random actions until the teacher/student become reasonably stable.
            # This avoids the policy collapsing early due to noisy pseudo labels.
            if rl_agent is None :
                img_mask, loss_mask = context_mask(img_a, args.mask_ratio)
            
            train_policy = iter_num >= args.rl_warmup_iters # 这里是一个判断逻辑--train_policy为bool
            if train_policy:
                action = rl_agent.choose_action(state, train=True)
            else:
                action = random.randrange(4)
            action_counts[action] += 1
            
            
            # ---- DEBUG: print RL behavior in terminal ----
            if iter_num % 100 == 0:
                eps = getattr(rl_agent, "epsilon", None)
                if eps is None and hasattr(rl_agent, "cfg"):
                    eps = getattr(rl_agent.cfg, "epsilon", None)
                print(f"[Iter {iter_num}] train_policy={train_policy} action={action} epsilon={eps}")
            
            if iter_num % 500 == 0:
                print("AMP scaler scale:", scaler.get_scale())


            # ================================
            # (3) Generate STRICT BCP masks from action
            #     IMPORTANT: guided mask must be derived from the SAME image it will be applied to.
            # ================================
            with torch.no_grad():
                # IMPORTANT FIX (minimal, high impact):
                # 1) Use TWO masks so the guided mask is always computed from the SAME image it will be applied to.
                #    - M_out is derived from uimg_a maps and applied to (uimg_a, img_a).
                #    - M_in  is derived from uimg_b maps and applied to (img_b, uimg_b).
                #    This avoids the 'A-derived mask applied on B' misalignment noise.
                # 2) For action=2 (teacher-student disagreement), we need student logits on BOTH uimg_a and uimg_b.
                student_logits_b = None
                if action == 2:
                    # Use eval-mode logits for stable disagreement maps (avoid BN/dropout noise).
                    model_was_training_mask = model.training
                    model.eval()
                    student_logits_b, _, _ = model(unimg_b)
                    if model_was_training_mask:
                        model.train()
                
                M_out = generate_mask_by_action(
                    action,
                    teacher_logits=teacher_logits_a,
                    student_logits=student_logits_a if action == 2 else None,
                    cfg=mask_cfg,
                )  # [B,H,W] in {0,1}
                M_in = generate_mask_by_action(
                    action,
                    teacher_logits=teacher_logits_b,
                    student_logits=student_logits_b if action == 2 else None,
                    cfg=mask_cfg,
                )  # [B,H,W] in {0,1}
            
                if iter_num % 200 == 0:
                    cover_out = M_out.mean().item()
                    cover_in  = M_in.mean().item()
                    print(
                        f"[Iter {iter_num:6d}] "
                        f"mask_cover_out={cover_out:.3f}, "
                        f"mask_cover_in={cover_in:.3f}"
                    )
                    
            # Convert masks to float and add channel dim for image mixing.
            M_out_img = M_out.unsqueeze(1).to(device=volume_batch.device, dtype=torch.float32)
            M_in_img  = M_in.unsqueeze(1).to(device=volume_batch.device, dtype=torch.float32)
            M_out = M_out.to(device=volume_batch.device, dtype=torch.float32)
            M_in  = M_in.to(device=volume_batch.device, dtype=torch.float32)
            
            consistency_weight = get_current_consistency_weight(iter_num // 150)
            
            # ================================
            # (4) BCP forward + supervised losses (STRICT alignment)
            #   - M_out: derived from unimg_a maps, applied to (img_a, unimg_a)
            #   - M_in : derived from unimg_b maps, applied to (unimg_b, img_b)
            # ================================
            mixl_img = img_a * M_out_img + unimg_a * (1 - M_out_img)
            mixu_img = unimg_b * M_in_img + img_b * (1 - M_in_img)
            
            mixl_lab = lab_a * M_out_img + plab_a * (1 - M_out_img)
            mixu_lab = plab_b * M_in_img + lab_b * (1 - M_in_img)
            
            outputs_l, _, _ = model(mixl_img)
            outputs_u, _, _ = model(mixu_img)
            
            # loss 必须与对应 mask/label 对齐
            loss_l = mix_loss(outputs_l, lab_a,  plab_a, M_out, u_weight=args.u_weight)
            loss_u = mix_loss(outputs_u, plab_b, lab_b,  M_in,  u_weight=args.u_weight, unlab=True)

            loss = loss_l + loss_u

            iter_num += 1
            writer.add_scalar('Self/consistency', consistency_weight, iter_num)
            writer.add_scalar('Self/loss_l', loss_l, iter_num)
            writer.add_scalar('Self/loss_u', loss_u, iter_num)
            writer.add_scalar('Self/loss_all', loss, iter_num)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if iter_num % 200 == 0:
                logging.info('iteration %d : loss: %03f, loss_l: %03f, loss_u: %03f'%(iter_num, loss, loss_l, loss_u))

            update_ema_variables(model, ema_model, 0.99)
            
            
            # ================================
            # (5) STRICT BCP reward: -(BCP loss + edge proxy)
            #     target must match outputs branch and mask used in forward
            # ================================
            with torch.no_grad():
                # outputs_l 对应 mixl： (img_a, unimg_a) + M_out
                target_l = (lab_a * M_out + plab_a * (1.0 - M_out)).long()

                # outputs_u 对应 mixu： (unimg_b, img_b) + M_in
                target_unl = (plab_b * M_in + lab_b * (1.0 - M_in)).long()

                edge_l = edge_proxy_penalty(outputs_l, target_l)
                edge_unl = edge_proxy_penalty(outputs_u, target_unl)
                edge_total = 0.5 * (edge_unl + edge_l)

                score = -float((loss.detach() + lambda_edge * edge_total).item())
                adv = score - ema_score
                ema_score = ema_momentum_reward * ema_score + (1.0 - ema_momentum_reward) * score
                reward = float(np.clip(adv, -reward_clip, reward_clip) * reward_scale)

                action_reward_sum[action] += reward
                
            
            if iter_num % 50 == 0:
                with torch.no_grad():
                    # 1) 验证 target_l/target_unl 与 mixl_lab/mixu_lab 的构造逻辑一致
                    # mixl_lab/mixu_lab 通常是 float，target 是 long，这里统一到 long 再比
                    target_l_chk   = (lab_a  * M_out + plab_a * (1.0 - M_out)).long()
                    target_unl_chk = (plab_b * M_in  + lab_b  * (1.0 - M_in )).long()

                    # 注意：mixl_lab/mixu_lab 你若保留了 *_img 版本（带 channel 维），这里相应 squeeze
                    mixl_lab_chk = (lab_a * M_out + plab_a * (1.0 - M_out)).long()
                    mixu_lab_chk = (plab_b * M_in + lab_b * (1.0 - M_in)).long()

                    # target 与自身重算应完全一致（用于防止你不小心又引用错变量）
                    assert (target_l == target_l_chk).all(),   "target_l mismatch: check M_out/lab_a/plab_a mapping"
                    assert (target_unl == target_unl_chk).all(), "target_unl mismatch: check M_in/lab_b/plab_b mapping"

                    # 2) 验证 mixl/mixu 的“图像混合方向”与 mask 是一致的（不是必要但很有用）
                    # 统计混合后，来自第一张图的比例（期望接近 mask_cover）
                    frac_from_first_l = (M_out.mean().item())
                    frac_from_first_u = (M_in.mean().item())

                    print(f"[Sanity] iter={iter_num} M_out_mean={frac_from_first_l:.3f} M_in_mean={frac_from_first_u:.3f}")


                # next state (optional; bandit uses immediate reward)
                # model_was_training2 = model.training
                # model.eval()
                # student_logits_a2, _, enc_feat_a2 = model(unimg_a)
                # pS2 = F.softmax(student_logits_a2, dim=1)
                # dis_mean2 = float((pT * (torch.log(torch.clamp(pT, 1e-6, 1.0)) - torch.log(torch.clamp(pS2, 1e-6, 1.0)))).sum(dim=1).mean().item())
                # next_state = extract_state_from_encoder_feature_3d(enc_feat_a2, ent_mean, dis_mean2)
                # if model_was_training2:
                #     model.train()
            # 关键：bandit 不用 next_state（update 里没用）。:contentReference[oaicite:2]{index=2}
            next_state = state  # 或 state.copy()
                    
            # RL memory/update
            rl_agent.store_transition(state, action, reward, next_state)
            rl_loss = rl_agent.update(batch_size=32) if train_policy else None  
            
            if iter_num % 200 == 0:
                rl_loss_str = "None" if rl_loss is None else f"{rl_loss:.4f}"
                print(
                    f"[Iter {iter_num}] "
                    f"reward={reward:.4f} | "
                    f"rl_loss_total={rl_loss_str}"
                )
            

             # change lr
            if iter_num % 2500 == 0:
                lr_ = base_lr * 0.1 ** (iter_num // 2500)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_

            if iter_num % 200 == 0:
                model.eval()
                dice_sample = test_3d_patch.var_all_case_LA(model, num_classes=num_classes, patch_size=patch_size, stride_xy=18, stride_z=4)
                if dice_sample > best_dice:
                    best_dice = round(dice_sample, 4)
                    save_mode_path = os.path.join(self_snapshot_path,  'iter_{}_dice_{}.pth'.format(iter_num, best_dice))
                    save_best_path = os.path.join(self_snapshot_path,'{}_best_model.pth'.format(args.model))
                    # save_net_opt(model, optimizer, save_mode_path)
                    # save_net_opt(model, optimizer, save_best_path)
                    torch.save(model.state_dict(), save_mode_path)
                    torch.save(model.state_dict(), save_best_path)
                    logging.info("save best model to {}".format(save_mode_path))
                writer.add_scalar('4_Var_dice/Dice', dice_sample, iter_num)
                writer.add_scalar('4_Var_dice/Best_dice', best_dice, iter_num)
                model.train()
            
            # if iter_num % 200 == 1:
            #     ins_width = 2
            #     B,C,H,W,D = outputs_l.size()
            #     snapshot_img = torch.zeros(size = (D, 3, 3*H + 3 * ins_width, W + ins_width), dtype = torch.float32)

            #     snapshot_img[:,:, H:H+ ins_width,:] = 1
            #     snapshot_img[:,:, 2*H + ins_width:2*H + 2*ins_width,:] = 1
            #     snapshot_img[:,:, 3*H + 2*ins_width:3*H + 3*ins_width,:] = 1
            #     snapshot_img[:,:, :,W:W+ins_width] = 1

            #     outputs_l_soft = F.softmax(outputs_l, dim=1)
            #     seg_out = outputs_l_soft[0,1,...].permute(2,0,1) # y
            #     target =  mixl_lab[0,...].permute(2,0,1)
            #     train_img = mixl_img[0,0,...].permute(2,0,1)

            #     snapshot_img[:, 0,:H,:W] = (train_img-torch.min(train_img))/(torch.max(train_img)-torch.min(train_img))
            #     snapshot_img[:, 1,:H,:W] = (train_img-torch.min(train_img))/(torch.max(train_img)-torch.min(train_img))
            #     snapshot_img[:, 2,:H,:W] = (train_img-torch.min(train_img))/(torch.max(train_img)-torch.min(train_img))

            #     snapshot_img[:, 0, H+ ins_width:2*H+ ins_width,:W] = target
            #     snapshot_img[:, 1, H+ ins_width:2*H+ ins_width,:W] = target
            #     snapshot_img[:, 2, H+ ins_width:2*H+ ins_width,:W] = target

            #     snapshot_img[:, 0, 2*H+ 2*ins_width:3*H+ 2*ins_width,:W] = seg_out
            #     snapshot_img[:, 1, 2*H+ 2*ins_width:3*H+ 2*ins_width,:W] = seg_out
            #     snapshot_img[:, 2, 2*H+ 2*ins_width:3*H+ 2*ins_width,:W] = seg_out
                
            #     writer.add_images('Epoch_%d_Iter_%d_labeled'% (epoch, iter_num), snapshot_img)

            #     outputs_u_soft = F.softmax(outputs_u, dim=1)
            #     seg_out = outputs_u_soft[0,1,...].permute(2,0,1) # y
            #     target =  mixu_lab[0,...].permute(2,0,1)
            #     train_img = mixu_img[0,0,...].permute(2,0,1)

            #     snapshot_img[:, 0,:H,:W] = (train_img-torch.min(train_img))/(torch.max(train_img)-torch.min(train_img))
            #     snapshot_img[:, 1,:H,:W] = (train_img-torch.min(train_img))/(torch.max(train_img)-torch.min(train_img))
            #     snapshot_img[:, 2,:H,:W] = (train_img-torch.min(train_img))/(torch.max(train_img)-torch.min(train_img))

            #     snapshot_img[:, 0, H+ ins_width:2*H+ ins_width,:W] = target
            #     snapshot_img[:, 1, H+ ins_width:2*H+ ins_width,:W] = target
            #     snapshot_img[:, 2, H+ ins_width:2*H+ ins_width,:W] = target

            #     snapshot_img[:, 0, 2*H+ 2*ins_width:3*H+ 2*ins_width,:W] = seg_out
            #     snapshot_img[:, 1, 2*H+ 2*ins_width:3*H+ 2*ins_width,:W] = seg_out
            #     snapshot_img[:, 2, 2*H+ 2*ins_width:3*H+ 2*ins_width,:W] = seg_out

            #     writer.add_images('Epoch_%d_Iter_%d_unlabel'% (epoch, iter_num), snapshot_img)

            # if iter_num >= self_max_iterations:
            #     break
            
            if rl_agent is not None:

                writer.add_scalar('RL/reward', reward, iter_num)
                writer.add_scalar('RL/action', action, iter_num)
                # writer.add_scalar('RL/rl_loss', rl_loss, iter_num)

                if iter_num % 200 == 0:
                    total = sum(action_counts.values()) + 1e-6
                    action_dist = {k: float(action_counts[k] / total) for k in action_counts}
                    avg_reward = {k: float(action_reward_sum[k] / max(1, action_counts[k])) for k in action_counts}
                    logging.info(f"[RL] iter={iter_num} action_dist={action_dist} avg_reward={avg_reward} rl_loss={rl_loss}")

                    logging.info('iteration %d : loss: %03f, loss_l: %03f, loss_u: %03f' %
                                (iter_num, loss, loss_l, loss_u))

        if iter_num >= self_max_iterations:
            iterator.close()
            break
    writer.close()


if __name__ == "__main__":
    ## make logger file
    pre_snapshot_path = "./model/BCP/LA_{}_{}_labeled/pre_train".format(args.exp, args.labelnum)
    self_snapshot_path = "./model/BCP/LA_{}_{}_labeled/self_train".format(args.exp, args.labelnum)
    print("Starting BCP training.")
    for snapshot_path in [pre_snapshot_path, self_snapshot_path]:
        if not os.path.exists(snapshot_path):
            os.makedirs(snapshot_path)
        if os.path.exists(snapshot_path + '/code'):
            shutil.rmtree(snapshot_path + '/code')
    shutil.copy('./code/Exchange_Mix_LA_BCP_train.py', self_snapshot_path)
    # -- Pre-Training
    logging.basicConfig(filename=pre_snapshot_path+"/log.txt", level=logging.INFO, format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    pre_train(args, pre_snapshot_path)
    # -- Self-training
    logging.basicConfig(filename=self_snapshot_path+"/log.txt", level=logging.INFO, format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    self_train(args, pre_snapshot_path, self_snapshot_path)

    
