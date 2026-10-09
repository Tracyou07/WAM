# Round02 方法契约 v0.2：官方加权损失下的两路径条件回归

2026-10-08。状态：供隔离原型实现与PM冻结审查；不代表集成通过或效果成立。替代v0.1，原版保存在archive_v01/。只改algorithm/。OpenWAM固定提交：4b3814a82268f3523df5fcdadcbeb2c021ac7737。

## 修正与研究定位

v0.1的整体有效坐标均值遗漏了官方timestep权重与逐token归一化。v0.2以官方 `_masked_action_flow_match_loss` 的逐样本分量定义分支能量，保留原权重、token/batch分母和float32计算。prior范围0.1–0.9、world/action系数1、普通AdamW均不变；没有新增温度。这是实现前发现的对齐修正，不是看结果后调参。

已核对11组论文及官方源码，另核对4篇变分近邻，见literature_code_audit.md/json。SFG、AdaShare、DynaShare及ConsMTL等已有共享结构、实例路由与私有参数干预先例。本方法是**加权条件Gaussian回归的有限混合surrogate**，不是新变分原理，也不是已建立的干净动作轨迹ELBO；新颖性与控制收益均未证实。

假说：在相同冻结范围和训练配方下，动作读取shared/private K/V的概率混合是否优于同容量确定性融合及强制private。posterior只表示对训练target的拟合责任，不直接度量world对action的更新伤害。

## 位置、所有权和可见性

- 必须运行dual_expert、current_block_coupling=decoupled_same_step、history_stream_visibility=video_only。检查effective配置和实际mask，不能根据文件名或program=joint推断。
- 官方配置视频30层、宽3072；动作30层、hidden_size2048。选择真实配对stack最后层zero-based29，不同时改14层。维度/bias以运行模块核验。
- attach_visual_tower把两专家block所有权转移至policy_variant.packed_block_stack.packed_blocks；原ModuleList清空并保留非owning执行视图。冻结审计在attach之后、optimizer/FSDP之前，检查Parameter identity及实际执行对象。
- 世界可训练phi仅最后层video_block.attn1.to_k/to_v的实际weight/bias；其他video、VAE、text全冻结。动作psi按同一受限基线范围训练。原官方YAML训练整个video backbone，不能直接认为满足此限制。
- private eta独立复制phi投影的值；复制且冻结norm_k，使用相同RoPE。不得共享Parameter/storage或重复注册optimizer。输入是该层投影之前的norm_hidden，不能用detach(world K/V)冒充private。
- z=0 private、z=1 shared，每样本/conditioning window一个；仅改变最后层动作query读取的视频K/V。video query永远使用world K/V，action self K/V、mask、text cross-attention、FFN不变。上游复用且不detach，最后动作分支枚举两次，world前向和loss一次。
- `_prepare_self_attention_inputs_native`目前不返回norm_hidden，infra需保留中间量或等价重算，不能从key反推。原packed block共用joint K/V，private臂应为action query单独构造K/V，不能把video所见K/V也替换。
- 原官方visibility函数9组prefix/singleton合成设置未出现video→action边，FULL历史与JOINT耦合负对照出现该边。完整模型冻结、cache和实际layout仍待验收，详见source_isolation_review.md。

prior只读真实已观测prefix的冻结VAE latent channel mean。observed mask必须来自官方prefix/layout；不读其他teacher-forced clean future token、action GT、残差或posterior。p=0.1+0.8*sigmoid(Linear(c0))，Linear零初始化，p初态0.5。缺prefix证据即失败。

## 官方分支能量与概率目标

输入pred_z,target:[B,T,D]；action_mask:[B,T,D]或None；w:[B,T]由同一个官方scheduler生成。target detach，MSE在float32计算；各臂共用noise、target、timesteps、mask、w。无mask时m=1。

```
n_it = max(sum_d m_itd, 1)
w_it = scheduler.training_weight(timesteps.flatten()).reshape(B,T)
ell_iz = (1/T) sum_t [w_it * sum_d m_itd*(float(f_z)-float(target.detach()))^2 / n_it]
official_action_loss(f_z) = mean_i ell_iz
a_itd = w_it*m_itd/(T*n_it)
```

全masked token贡献0但仍计入T，全无监督样本贡献0但仍计入B。不得替换成有效token数或有效坐标总数。w/m固定且与参数、route无关，有限非负。a=0的坐标从回归密度监督集合排除；样本全部a=0使用空乘积密度1，因此ell=0、q=p、action梯度0，并记录empty比例。负或非有限权重立即拒绝。

对每个a>0的坐标定义独立Gaussian N(target;f_z,1/(2*a))；NLL为ell加与参数和z无关的常数。二值mask下方差为T*n_it/(2*w_it)，替代v0.1的D_i/2。省略常数用于训练，不能将此日志值当作跨维度可比的绝对概率。

```
log_joint_iz = log prior_iz - ell_iz
q_i = softmax(log_joint_i)       # 精确训练后验，可见训练target
La_i = -logsumexp(log_joint_i)
total_loss = official_world_loss + mean_i La_i
```

action_loss_weight=latent_loss_weight=1。world保持原官方target/mask/weight/reduction，不另行重写。没有temperature、beta-KL、entropy、balance、branch auxiliary loss。

对任意q，F(q)=sum_z q_z*ell_z+KL(q||prior)>=La；精确q取等号。直接优化logsumexp；若用F(q)，须用精确q.detach并验证相同梯度。phi的action梯度只来自q_shared加权shared臂，eta来自private臂，psi来自两臂；此结论需完整模型private→phi零导数验收。

相同分支f0=f1=f时，mixture loss及对共同f/上游参数的总梯度等于官方loss，q=p。独立K/V副本各自梯度按p或1-p分配；不能要求某一个副本等于原生完整梯度。

## 接口与数值验证

参考实现：operator_v02.py中的route_objective_weighted。

```
route_objective_weighted(pred_private, pred_shared, target,
                         action_mask, timestep_weight, prior_shared)
 -> loss, per_sample_nll[B], branch_weighted_mse[B,2],
    posterior[B,2], kl[B], coefficient[B,T,D]
```

caller先用官方scheduler生成w，不在分支内重复采样；branch日志标为weighted token mean，不误标raw action MSE。与infra/loss_reduction_review.md公式一致。

反例：B=1,T=2,D=2；预测[[1,99],[2,2]]、target全0、mask[[1,0],[1,1]]、w全1。官方token均值2.5，v0.1有效坐标总均值3。无timestep差时v0.1也不满足parity。

operator_v02_results.json记录7项CPU通过：未改写官方loss函数的值/预测梯度parity；零权重/空token/空样本；weighted ELBO与detach-q梯度恒等；上述反例；官方visibility核9组；visibility两个负对照；FAMO原模块梯度及一次权重更新。首次harness遗漏原源码常量依赖，补齐后通过，未改作者函数。没有完整模型或benchmark复现。

## 固定灵敏度：不临时加温度

Delta=ell_private-ell_shared，则q_shared=sigmoid(logit(p_shared)+Delta)。对任意p，abs(q-p)<=abs(Delta)/4。Delta单位是官方加权token均值。

| p=0.5的Delta | q_shared |
|---:|---:|
| 0 | 0.500000 |
| 0.001 | 0.500250 |
| 0.01 | 0.502500 |
| 0.04 | 0.509999 |
| 0.1 | 0.524979 |
| 0.405465 | 0.600000 |
| 1 | 0.731059 |
| 2.197225 | 0.900000 |

预先固定诊断：abs(Delta)<=0.04的样本必有abs(q-p)<=0.01；报告覆盖率和Delta分位数。这是算术灵敏度界，不是效果显著性阈值。p=.1到q=.9需要Delta约4.394449。允许实测小尺度下q≈p和prior梯度弱成为失败结果，不事后调temperature/KL。prior边界也不能防posterior饱和或分支训练不足。

## 四臂和第五臂状态

首轮四臂：native_joint（名称沿用，实际为同decoupled mask和受限冻结范围的原生单路）；same_capacity_deterministic_kv_blend；variational_sharing；forced_private_world_gradient_off。blend在完成norm/RoPE的K/V上按prior凸组合，只算一次官方加权MSE；强制private保留world监督，仅使phi不收到action梯度。各臂参数量、初始化、步数和计算量单独报告，不能把较少参数的原生臂也叫同容量。

第五臂区分共享位置与普通生成端mixture，**扩大pilot前必须冻结**。廉价可复用位置：VideoConditionedActionExpert.post_dit的norm/modulation之后、action_proj_out处。共享最终h，独立复制两个线性输出头，f_z=W_z*h+b_z；两臂都读取同一shared world K/V，复用prior、weighted evidence及整段固定route接口。两臂都可能回传phi，因此是生成端mixture对照。

但该双头不满足同容量：action hidden2048、action_dim7时额外head为14,343参数；private视频K/V两个3072方阵约18.88M（精确bias/numel须运行时核验）。不能用闲置参数或巨型输出MLP凑数。撤回v0.1中“同容量output mixture已经定义好”的表述。四臂可做正确性smoke，第五臂容量匹配仍未冻结；缺它时不得声称已排除一般mixture容量解释或据此扩大正式性能结论。

## 推理与完整验收

部署只用causal prior，以单独route RNG每action ODE/chunk采样一次并固定。禁止使用target posterior、逐步重采样或无声明改MAP/均值速度。配对固定action noise与route uniform；两RNG分离并保存。ordinary AdamW一套状态，参数无重复注册；保存optimizer、scheduler、数据位置和RNG。

待完整模型验收：层数/冻结集合、独立storage、private→phi零导数且shared非零、真实prefix无未来泄漏、torch/flex语义一致、官方checkpoint前向parity、optimizer/data/RNG恢复、BF16/FSDP。当前数学参考与原源码小函数测试不替代这些项。

效果验证使用相同incoming optimizer状态，先测world加入前后实际action/world训练probe变化，再看配对留出控制任务；probe不作正式测试集，不用测试结果挑超参。训练MSE、q偏好与cosine不等于控制收益。若不优于blend或强制private，应否定当前增益解释。
