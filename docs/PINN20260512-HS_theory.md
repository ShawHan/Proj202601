# PINN20260512-HS 方法理论说明

## 1. 方法定位

`PINN20260512-HS` 是一个面向 PECVD-TEOS/O2 场景的物理驱动混合模型。模型首先依据 Kim 2000 对 TEOS 等离子体沉积 SiO2 的动力学分析，构造从工艺输入到气相物种、表面羟基覆盖、侧链脱水交联、沉积速率和晶圆厚度统计量的可微物理流程；然后只在不可观测的反应常数、等离子体活性、输运闭合项和残差项上引入 AI 拟合。

模型输入为：

$$
x=\{P,\ Q_{O_2},\ Q_{\mathrm{TEOS}},\ Q_{\mathrm{He}},\ t,\ s,\ T,\ P_{\mathrm{rf}}\}
$$

模型输出为：

$$
y=\{H,\ H_{\max},\ H_{\min},\ D,\ U_{\mathrm{range}},\ U_{1\sigma},\ RI,\ \sigma\}
$$

其中 $H$ 为平均厚度，$D$ 为沉积速率，$RI$ 为折射率，$\sigma$ 为应力。

## 2. 工艺量归一化与停留时间近似

总流量定义为：

$$
Q_{\mathrm{tot}}=Q_{O_2}+Q_{\mathrm{TEOS}}+Q_{\mathrm{He}}
$$

对工艺变量做参考尺度归一化：

$$
\hat{P}=\frac{P}{P_0},\quad
\hat{s}=\frac{s}{s_0},\quad
\hat{t}=\frac{t}{t_0},\quad
\hat{P}_{\mathrm{rf}}=\frac{P_{\mathrm{rf}}}{P_{\mathrm{rf},0}}
$$

用压力、极板间距和总流量构造有效停留时间因子：

$$
\tau_{\mathrm{res}}
=
\hat{P}\hat{s}\frac{Q_{\mathrm{tot},0}}{Q_{\mathrm{tot}}}
$$

TEOS 与氧气的有效进料项为：

$$
\Phi_{\mathrm{TEOS}}
=
\frac{Q_{\mathrm{TEOS}}}{Q_{\mathrm{TEOS},0}}\hat{P}\tau_{\mathrm{res}}
$$

$$
\Phi_{O_2}
=
\frac{Q_{O_2}}{Q_{O_2,0}}\hat{P}\tau_{\mathrm{res}}
$$

温度中心化变量为：

$$
\Delta T
=
\frac{(T+273.15)-(T_0+273.15)}{50}
$$

## 3. TEOS 气相解离链

Kim 论文将 TEOS 在等离子体中逐步解离为一系列含羟基前驱物。模型采用如下链式结构近似：

$$
\mathrm{TEOS}\rightarrow P_1\rightarrow P_2\rightarrow P_3\rightarrow P_4
$$

其中 $P_i$ 对应不同程度解离/羟基化的 TEOS 前驱物。未充分解离的母体项为：

$$
P_0
=
\frac{\Phi_{\mathrm{TEOS}}}{1+x_1\hat{P}_{\mathrm{rf}}}
$$

逐级解离物种浓度近似为：

$$
P_1
=
\frac{
\Phi_{\mathrm{TEOS}}\hat{P}_{\mathrm{rf}}
}{
(\hat{P}_{\mathrm{rf}}+x_1)(\hat{P}_{\mathrm{rf}}+x_2)
}
m_1
$$

$$
P_2
=
\frac{
\Phi_{\mathrm{TEOS}}\hat{P}_{\mathrm{rf}}^2
}{
(\hat{P}_{\mathrm{rf}}+x_1)(\hat{P}_{\mathrm{rf}}+x_2)(\hat{P}_{\mathrm{rf}}+x_3)
}
m_2
$$

$$
P_3
=
\frac{
\Phi_{\mathrm{TEOS}}\hat{P}_{\mathrm{rf}}^3
}{
(\hat{P}_{\mathrm{rf}}+x_1)(\hat{P}_{\mathrm{rf}}+x_2)(\hat{P}_{\mathrm{rf}}+x_3)(\hat{P}_{\mathrm{rf}}+x_4)
}
m_3
$$

$$
P_4
=
\frac{
\Phi_{\mathrm{TEOS}}\hat{P}_{\mathrm{rf}}^4
}{
(\hat{P}_{\mathrm{rf}}+x_1)(\hat{P}_{\mathrm{rf}}+x_2)(\hat{P}_{\mathrm{rf}}+x_3)(\hat{P}_{\mathrm{rf}}+x_4)(\hat{P}_{\mathrm{rf}}+x_5)
}
m_4
$$

其中 $x_i>0$ 为可学习的有效寿命/反应竞争参数，$m_i$ 为小幅 AI 闭合修正。

## 4. 氧自由基与氧离子密度

Kim 论文中氧自由基和分子氧离子随射频功率增加，并受损失寿命限制。模型采用：

$$
n_O
=
\frac{
2\Phi_{O_2}\hat{P}_{\mathrm{rf}}
}{
\hat{P}_{\mathrm{rf}}+x_O
}
m_O
$$

$$
n_{\mathrm{ion}}
=
\frac{
\Phi_{O_2}\hat{P}_{\mathrm{rf}}
}{
\hat{P}_{\mathrm{rf}}+x_{\mathrm{ion}}
}
m_{\mathrm{ion}}
$$

其中 $n_O$ 表示氧自由基密度，$n_{\mathrm{ion}}$ 表示氧离子密度。

## 5. 前驱体表面通量

每类前驱物的表面反应速率常数使用温度修正：

$$
k_i(T)
=
\mathrm{softplus}(a_i)
\exp\left(\mathrm{clip}(b_i\Delta T,-2,2)\right)
$$

总前驱体通量为：

$$
\Phi_P
=
k_1P_1+k_2P_2+k_3P_3+k_4P_4
$$

## 6. 表面羟基覆盖率

Kim 论文强调沉积受表面羟基覆盖控制。模型将氧自由基、二阶氧自由基项和氧离子共同视为羟基生成驱动力：

$$
\Omega
=
k_{O,1}n_O+k_{O,2}n_O^2+k_{\mathrm{ion}}n_{\mathrm{ion}}
$$

表面羟基覆盖率为：

$$
u_{\mathrm{OH}}
=
\frac{
\Omega
}{
\Omega+k_{\mathrm{ads}}\Phi_P+\varepsilon
}
$$

其中 $k_{\mathrm{ads}}\Phi_P$ 表示前驱体吸附消耗表面活性位的竞争项。

## 7. Si-O 键保留与输运修正

Kim 论文指出高功率条件下 Si-O 键解离会影响沉积。模型使用如下保留因子：

$$
S_{\mathrm{SiO}}
=
\frac{1}{1+x_{\mathrm{SiO}}\hat{P}_{\mathrm{rf}}}
$$

输运项综合压力、极板间距和 He 稀释效应：

$$
G_{\mathrm{tr}}
=
\exp[-a_s(\hat{s}-1)]
\hat{P}^{a_P}
\exp[-a_{\mathrm{He}}(f_{\mathrm{He}}-0.5)]
$$

其中：

$$
f_{\mathrm{He}}
=
\frac{Q_{\mathrm{He}}}{Q_{\mathrm{tot}}}
$$

时间和功率修正为：

$$
G_t=\hat{t}^{a_t}
$$

$$
G_P=\hat{P}_{\mathrm{rf}}^{a_{\mathrm{rf}}}
$$

## 8. 沉积速率

基础沉积速率由前驱体通量、羟基覆盖、Si-O 保留、输运、时间和功率项共同决定：

$$
D_{\mathrm{phys}}
=
A
\Phi_P
u_{\mathrm{OH}}
S_{\mathrm{SiO}}
G_{\mathrm{tr}}
G_t
G_P
m_D
$$

模型再加入一个有界 AI 速率残差：

$$
\delta_D
=
\rho \tanh(h_{\theta}(z_D))
$$

最终沉积速率为：

$$
D
=
D_{\mathrm{phys}}\exp(\delta_D)
$$

其中 $z_D$ 不是原始输入，而是 Kim 物理状态：

$$
z_D
=
\{
P_0,P_1,P_2,P_3+P_4,n_O,n_{\mathrm{ion}},
u_{\mathrm{OH}},S_{\mathrm{SiO}},G_{\mathrm{tr}},
\tau_{\mathrm{res}},\hat{t},Q_{O_2}/Q_{\mathrm{TEOS}},
s_{\mathrm{ox}},j_G,\Phi_P,\Delta T,f_{\mathrm{He}}
\}
$$

## 9. 厚度预测

平均厚度由沉积速率和沉积时间给出：

$$
H
=
D\frac{t}{60}
$$

## 10. 侧链氧化、脱水与交联

侧链氧化强度由氧自由基和氧离子驱动：

$$
s_{\mathrm{ox,raw}}
=
k_{s,O}n_O^2+k_{s,\mathrm{ion}}n_{\mathrm{ion}}
$$

$$
s_{\mathrm{ox}}
=
0.49
\frac{
s_{\mathrm{ox,raw}}
}{
1+s_{\mathrm{ox,raw}}
}
$$

脱水强度为：

$$
k_{\mathrm{dehyd}}
=
0.49
\sigma_g
\left(
b_0+b_T\Delta T+b_{\mathrm{ion}}\log(1+n_{\mathrm{ion}})
\right)
$$

其中 $\sigma_g(\cdot)$ 为 sigmoid 函数。

初始侧链中有机基团和羟基比例为：

$$
j_{R,1}
=
\frac{
2k_1P_1+2k_2P_2+k_3P_3
}{
2\Phi_P
}
$$

$$
j_{\mathrm{OH},1}
=
\frac{
k_3P_3+2k_4P_4
}{
2\Phi_P
}
$$

经过氧化后的比例为：

$$
j_{R,2}
=
j_{R,1}(1-2s_{\mathrm{ox}})
$$

$$
j_{\mathrm{OH},2}
=
j_{R,1}s_{\mathrm{ox}}
+
j_{\mathrm{OH},1}(1-2k_{\mathrm{dehyd}})
$$

经过脱水后的比例为：

$$
j_{R,3}=j_{R,2}
$$

$$
j_{\mathrm{OH},3}
=
j_{\mathrm{OH},2}(1-2k_{\mathrm{dehyd}})
$$

完整交联比例为：

$$
j_{G,\mathrm{full}}
=
\frac{1}{2}
\left(
1-j_{\mathrm{OH},3}-j_{R,3}
\right)
$$

简化交联比例为：

$$
j_{G,\mathrm{simple}}
=
k_{\mathrm{dehyd}}s_{\mathrm{ox}}
$$

最终交联比例为：

$$
j_G
=
0.65j_{G,\mathrm{full}}
+
0.35j_{G,\mathrm{simple}}
$$

## 11. 膜致密化、折射率与应力

沉积进入膜网络的有效速率为：

$$
C
=
D(1+2j_G)
$$

膜致密化状态为：

$$
\eta
=
\sigma_g
\left(
c_1j_G
+c_2\log(1+n_{\mathrm{ion}})
+c_3\Delta T
-c_4(1-u_{\mathrm{OH}})
\right)
$$

折射率和应力由交联、羟基覆盖、离子比例、氧自由基比例、致密化和沉积速率共同预测：

$$
RI
=
RI_0
+
\Delta RI
\tanh(f_{RI}(j_G,u_{\mathrm{OH}},n_{\mathrm{ion}},n_O,\eta,C,\Delta T))
$$

$$
\sigma
=
\sigma_0
+
\Delta\sigma
\tanh(f_{\sigma}(j_G,u_{\mathrm{OH}},n_{\mathrm{ion}},n_O,\eta,C,\Delta T))
$$

## 12. 晶圆厚度统计量

数据集中只有晶圆采样点统计量，没有完整空间坐标。因此模型不直接拟合每个采样点，而是构造一个归一化径向厚度轮廓：

$$
h(r)
=
H
\frac{
S(r)
}{
\mathbb{E}_r[S(r)]
}
$$

其中：

$$
S(r)
=
\max
\left(
1+\sum_m c_mB_m(r),\ 0.30
\right)
$$

$B_m(r)$ 为径向基函数，$c_m$ 由氧自由基、氧离子、停留时间、功率、极板间距和温度预测。

厚度统计量为：

$$
H_{\max}
=
\max_r h(r)
$$

$$
H_{\min}
=
\min_r h(r)
$$

$$
U_{\mathrm{range}}
=
\frac{
H_{\max}-H_{\min}
}{
2H
}
\times 100
$$

$$
U_{1\sigma}
=
\frac{
\mathrm{std}_r(h(r))
}{
H
}
\times 100
$$

最终还使用轻量一致性修正：

$$
H_{\max}
\leftarrow
(1-\gamma)H_{\max}
+
\gamma
\left(
H_{\min}
+2H\frac{U_{\mathrm{range}}}{100}
\right)
$$

当前默认值为：

$$
\gamma=0.35
$$

## 13. AI 闭合项

模型中的 AI 不直接替代物理公式，而是只用于补偿不可观测的微观过程。

第一类 AI 是物理链条中的有界乘法闭合项：

$$
m
=
\exp
\left(
\alpha\tanh(g_{\theta}(z_{\mathrm{phys}}))
\right)
$$

其中 $z_{\mathrm{phys}}$ 为归一化后的物理状态：

$$
z_{\mathrm{phys}}
=
\{
\Phi_{\mathrm{TEOS}},
\Phi_{O_2},
Q_{O_2}/Q_{\mathrm{TEOS}},
\tau_{\mathrm{res}},
\hat{P}_{\mathrm{rf}},
\hat{t},
\Delta T,
\hat{s},
f_{\mathrm{He}},
\hat{Q}_{\mathrm{He}}
\}
$$

第二类 AI 是沉积速率残差：

$$
D
=
D_{\mathrm{phys}}\exp(\delta_D)
$$

$$
\delta_D
=
\rho\tanh(h_{\theta}(z_D))
$$

第三类 AI 是最终 Kim 状态校准器。它不接收原始输入，而接收物理中间状态：

$$
\psi
=
\{
\hat{y}_{\mathrm{PINN}},
P_0,\ldots,P_4,
n_O,n_{\mathrm{ion}},
\Phi_P,\Omega,u_{\mathrm{OH}},
S_{\mathrm{SiO}},
G_{\mathrm{tr}},
j_G,\eta,C,
c_1,\ldots,c_M,
\tau_{\mathrm{res}},
Q_{O_2}/Q_{\mathrm{TEOS}},
\Delta T
\}
$$

direct 校准模式下：

$$
\tilde{y}_j
=
\mathcal{C}_j(\psi)
$$

$$
\hat{y}_j
=
(1-\beta)\hat{y}_{j,\mathrm{PINN}}
+
\beta\tilde{y}_j
$$

当前默认值为：

$$
\beta=1
$$

因此最终输出由 Kim 物理状态驱动，而不是由原始输入黑箱映射得到。

## 14. 监督信号与损失函数

监督目标为数据集中的八个输出：

$$
y
=
\{H,H_{\max},H_{\min},D,U_{\mathrm{range}},U_{1\sigma},RI,\sigma\}
$$

对厚度、沉积速率、均匀性、折射率和应力分别计算监督损失：

$$
\mathcal{L}_{\mathrm{thk}}
=
\mathrm{Huber}
\left(
\frac{
\hat{y}_{H,H_{\max},H_{\min}}
-y_{H,H_{\max},H_{\min}}
}{
s_{H,H_{\max},H_{\min}}
}
\right)
$$

$$
\mathcal{L}_{\log \mathrm{thk}}
=
\mathrm{Huber}
\left(
\log(1+\hat{y}_{H,H_{\max},H_{\min}})
-
\log(1+y_{H,H_{\max},H_{\min}})
\right)
$$

$$
\mathcal{L}_{D}
=
\mathrm{Huber}
\left(
\frac{\hat{D}-D}{s_D}
\right)
$$

$$
\mathcal{L}_{\log D}
=
\mathrm{Huber}
\left(
\log(1+\hat{D})-\log(1+D)
\right)
$$

$$
\mathcal{L}_{U}
=
\mathrm{Huber}
\left(
\frac{
\hat{y}_{U_{\mathrm{range}},U_{1\sigma}}
-y_{U_{\mathrm{range}},U_{1\sigma}}
}{
s_{U_{\mathrm{range}},U_{1\sigma}}
}
\right)
$$

$$
\mathcal{L}_{RI}
=
\mathrm{Huber}
\left(
\frac{\widehat{RI}-RI}{s_{RI}}
\right)
$$

$$
\mathcal{L}_{\sigma}
=
\mathrm{Huber}
\left(
\frac{\hat{\sigma}-\sigma}{s_{\sigma}}
\right)
$$

物理正则项包括 AI 闭合幅度、速率残差、径向轮廓平滑和 Kim 状态约束：

$$
\mathcal{L}_{\mathrm{closure}}
=
\|g_{\theta}(z_{\mathrm{phys}})\|_2^2
$$

$$
\mathcal{L}_{\mathrm{rate\ residual}}
=
\|\delta_D\|_2^2
$$

$$
\mathcal{L}_{\mathrm{profile}}
=
\|c\|_2^2
$$

$$
\mathcal{L}_{\mathrm{Kim}}
=
\mathcal{R}(u_{\mathrm{OH}},j_G,S_{\mathrm{SiO}},\Phi_P)
$$

总损失为：

$$
\begin{aligned}
\mathcal{L}
=&
\lambda_{\mathrm{thk}}\mathcal{L}_{\mathrm{thk}}
+\lambda_{\log \mathrm{thk}}\mathcal{L}_{\log \mathrm{thk}}
+\lambda_D\mathcal{L}_D
+\lambda_{\log D}\mathcal{L}_{\log D} \\
&
+\lambda_U\mathcal{L}_U
+\lambda_{RI}\mathcal{L}_{RI}
+\lambda_{\sigma}\mathcal{L}_{\sigma}
+\lambda_{\mathrm{closure}}\mathcal{L}_{\mathrm{closure}} \\
&
+\lambda_{\mathrm{rate\ residual}}\mathcal{L}_{\mathrm{rate\ residual}}
+\lambda_{\mathrm{profile}}\mathcal{L}_{\mathrm{profile}}
+\lambda_{\mathrm{Kim}}\mathcal{L}_{\mathrm{Kim}}
\end{aligned}
$$

## 15. 总体流程

整体预测流程可写为：

$$
x
\rightarrow
(\Phi_{\mathrm{TEOS}},\Phi_{O_2},\tau_{\mathrm{res}})
\rightarrow
(P_0,\ldots,P_4,n_O,n_{\mathrm{ion}})
\rightarrow
(u_{\mathrm{OH}},\Phi_P,j_G,\eta)
\rightarrow
(D,H,h(r))
\rightarrow
y
$$

其中 AI 只参与：

$$
\text{unknown plasma/surface closure}
\quad
\text{and}
\quad
\text{Kim-state calibration}
$$

而不是直接学习：

$$
x\rightarrow y
$$

这使模型保持 PECVD 反应过程的可解释性，同时允许数据对不可测微观量进行补偿。
