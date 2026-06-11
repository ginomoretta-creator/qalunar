%{

Author: Enrico Schiassi
Modified by Gino Moretta — QUBO v2: Tikhonov + SVD reduction + per-variable encoding

Problem: Energy Optimal Rendezvous In Relative Motion (Clohessy-Wiltshire
Dynamics) with improved QUBO formulation.

Three improvements over v1:
  (1) Tikhonov regularization:  min ||A*xi - B||^2 + lambda*||xi||^2
  (2) SVD dimensionality reduction:  xi = V_r * alpha
  (3) Per-variable encoding ranges:

%}
%%

clearvars -except L p; close all; clc;
format long
rng('default')      % same seed as original — ensures identical A, B

%% Physical setup (identical to original)

mu    = 3.986004418*(10^14);
r_csp = 7500*(10^3);
w     = sqrt(mu/(r_csp^3));

M_cw = (1/w^2)*[3*w^2 0 0; 0 0 0; 0 0 -w^2];
MT   = M_cw';
N_cw = (1/w)*[0 2*w 0; -2*w 0 0; 0 0 0];
NT   = N_cw';

ToF = 2000*w;

r0 = [ 7047;  5136;  5013 ] / r_csp;
v0 = [ -2.4; -13.7;  4.08 ] / (r_csp*w);
rf = [ 0; 0; 0 ] / r_csp;
vf = [ 0; 0; 0 ] / (r_csp*w);

%% Input parameters

type_trainingPoints = 1;
n  = 20;
if ~exist('L','var'), L = 80; end
nt = 200;
type_activation = 2;

LBw = -3;  UBw = 3;
LBb = -3;  UBb = 3;

%% Training points

t0 = 0;  tf = ToF;
z0 = -1; zf =  1;

switch type_trainingPoints
    case 1, z = linspace(z0,zf,n)';
    case 2, z = z0 + (zf-z0)*rand(n,1); z(1) = z0; z(end) = zf;
end

c = (z(end)-z(1))/(tf-t0);  c2 = c^2;
t_train = t0 + (1/c)*(z - z(1));

%% Boundary values

r01 = r0(1); r02 = r0(2); r03 = r0(3);
v01 = v0(1); v02 = v0(2); v03 = v0(3);
rf1 = rf(1); rf2 = rf(2); rf3 = rf(3);
vf1 = vf(1); vf2 = vf(2); vf3 = vf(3);

%% Switching functions

dz = z(end)-z(1);  dz2 = dz^2;  dz3 = dz^3;

om1  =  1 + 2*(z-z(1)).^3/dz3 - 3*(z-z(1)).^2/dz2;
om2  = -2*(z-z(1)).^3/dz3 + 3*(z-z(1)).^2/dz2;
om3  = (z-z(1)) + (z-z(1)).^3/dz2 - 2*(z-z(1)).^2/dz;
om4  = (z-z(1)).^3/dz2 - (z-z(1)).^2/dz;

om1d =  6*(z-z(1)).^2/dz3 - 6*(z-z(1))/dz2;
om2d = -6*(z-z(1)).^2/dz3 + 6*(z-z(1))/dz2;
om3d =  1 + 3*(z-z(1)).^2/dz2 - 4*(z-z(1))/dz;
om4d =  3*(z-z(1)).^2/dz2 - 2*(z-z(1))/dz;

om1dd =  12*(z-z(1))/dz3 - 6/dz2;
om2dd = -12*(z-z(1))/dz3 + 6/dz2;
om3dd =   6*(z-z(1))/dz2 - 4/dz;
om4dd =   6*(z-z(1))/dz2 - 2/dz;

%% ELM

weight = LBw + (UBw-LBw)*rand(L,1);
bias   = LBb + (UBb-LBb)*rand(L,1);

h   = zeros(n,L);
hd  = zeros(n,L);
hdd = zeros(n,L);

for i = 1:n
    for j = 1:L
        [h(i,j), hd(i,j), hdd(i,j)] = act(z(i), weight(j), bias(j), type_activation);
    end
end

h0  = h(1,:);   hf  = h(end,:);
hd0 = hd(1,:);  hdf = hd(end,:);

%% Constrained expressions

F   = h   - om1.*h0  - om2.*hf  - om3.*hd0  - om4.*hdf;
Fd  = c  *(hd  - om1d.*h0  - om2d.*hf  - om3d.*hd0  - om4d.*hdf);
Fdd = c2 *(hdd - om1dd.*h0 - om2dd.*hf - om3dd.*hd0 - om4dd.*hdf);

C1   = om1.*r01 + om2.*rf1 + (1/c)*(om3.*v01 + om4.*vf1);
C1d  = c  *(om1d.*r01  + om2d.*rf1  + (1/c)*(om3d.*v01  + om4d.*vf1));
C1dd = c2 *(om1dd.*r01 + om2dd.*rf1 + (1/c)*(om3dd.*v01 + om4dd.*vf1));

C2   = om1.*r02 + om2.*rf2 + (1/c)*(om3.*v02 + om4.*vf2);
C2d  = c  *(om1d.*r02  + om2d.*rf2  + (1/c)*(om3d.*v02  + om4d.*vf2));
C2dd = c2 *(om1dd.*r02 + om2dd.*rf2 + (1/c)*(om3dd.*v02 + om4dd.*vf2));

C3   = om1.*r03 + om2.*rf3 + (1/c)*(om3.*v03 + om4.*vf3);
C3d  = c  *(om1d.*r03  + om2d.*rf3  + (1/c)*(om3d.*v03  + om4d.*vf3));
C3dd = c2 *(om1dd.*r03 + om2dd.*rf3 + (1/c)*(om3dd.*v03 + om4dd.*vf3));

H  = h;
Hd = c*hd;

%% Build linear system A_sys * xi = B_sys   (9n x 9L)

z  = zeros(n,L);
zz = zeros(6*n,1);

A1 = [Fdd-M_cw(1,1).*F,  -N_cw(1,2)*Fd,              z,  z,  z,  z;
      -N_cw(2,1)*Fd,       Fdd,                        z,  z,  z,  z;
       z,                   z,   Fdd-M_cw(3,3).*F,     z,  z,  z;
       z,                   z,                          z, Hd,  z,  z;
       z,                   z,                          z,  z, Hd,  z;
       z,                   z,                          z,  z,  z, Hd;
       z,                   z,                          z,  H,  z,  z;
       z,                   z,                          z,  z,  H,  z;
       z,                   z,                          z,  z,  z,  H];

A2 = [H,              z,             z;
      z,              H,             z;
      z,              z,             H;
      MT(1,1).*H,     z,             z;
      z,              z,             z;
      z,              z,  MT(3,3).*H;
      Hd, NT(1,2).*H,               z;
      NT(2,1)*H,     Hd,             z;
      z,              z,            Hd];

A_sys = [A1, A2];

B_sys = [-C1dd + 3*C1 + 2*C2d;
         -C2dd - 2*C1d;
         -C3dd - C3;
          zz];

N_vars = 9*L;
fprintf('\n=== LINEAR SYSTEM ===\n');
fprintf('  A_sys size : %d x %d\n', size(A_sys,1), size(A_sys,2));
fprintf('  N_vars     : %d\n', N_vars);

%% STEP 1 — TIKHONOV REGULARIZATION

% min ||A*xi - B||^2 + lambda*||xi||^2
% Equivalent to: min ||A_aug*xi - B_aug||^2
%   A_aug = [A; sqrt(lambda)*I]    (overdetermined: unique solution)
%   B_aug = [B; 0]

s_vals = svd(A_sys);
lambda_tik = 1e-6 * s_vals(1)^2;

A_aug = [A_sys; sqrt(lambda_tik) * eye(N_vars)];
B_aug = [B_sys; zeros(N_vars, 1)];

fprintf('\n=== TIKHONOV REGULARIZATION ===\n');
fprintf('  lambda          : %.4e\n', lambda_tik);
fprintf('  sigma_max(A)    : %.4e\n', s_vals(1));
fprintf('  sigma_min(A)    : %.4e\n', s_vals(end));
fprintf('  cond(A)         : %.4e\n', s_vals(1)/s_vals(end));
fprintf('  A_aug size      : %d x %d\n', size(A_aug,1), size(A_aug,2));


%% STEP 2 — SVD DIMENSIONALITY REDUCTION

% Compute SVD of A_aug, determine effective rank, reduce variables.

[U_aug, S_aug, V_aug] = svd(A_aug, 'econ');
s_diag = diag(S_aug);

% Effective rank: keep singular values above tau * sigma_max
tau_svd = 1e-3;  % threshold relative to largest SV
r = sum(s_diag > tau_svd * s_diag(1));

V_r   = V_aug(:, 1:r);        % 720 x r  (basis for xi)
U_r   = U_aug(:, 1:r);        % (180+720) x r
S_r   = S_aug(1:r, 1:r);      % r x r diagonal

A_red = A_aug * V_r;           % (180+720) x r  (reduced system matrix)

% Tikhonov solution in reduced coordinates
% xi_tik = V_r * alpha_tik  where  alpha_tik = S_r \ (U_r' * B_aug)
alpha_tik = S_r \ (U_r' * B_aug);
xi_tik    = V_r * alpha_tik;

% Validate: how well does the Tikhonov solution satisfy the original ODEs?
loss_tik = A_sys * xi_tik - B_sys;

fprintf('\n=== SVD REDUCTION ===\n');
fprintf('  tau_svd          : %.1e\n', tau_svd);
fprintf('  effective rank r : %d  (of %d)\n', r, N_vars);
fprintf('  reduction ratio  : %.1fx fewer variables\n', N_vars/r);
fprintf('  A_red size       : %d x %d\n', size(A_red,1), size(A_red,2));
fprintf('  Tikhonov residual: max|A*xi_tik - B| = %.4e\n', max(abs(loss_tik)));
fprintf('  ||xi_tik||       : %.6f\n', norm(xi_tik));


%% STEP 3 — PER-VARIABLE BINARY ENCODING

% Each reduced coordinate alpha_i gets its own encoding range [-R_i, +R_i]
% centered on the Tikhonov estimate alpha_tik(i).
%   - Center = alpha_tik (from SVD-based Tikhonov solve above)
%   - Spread = margin * (|alpha_tik_i| + sigma_i * ||residual||)
%     where sigma_i = 1/s_diag(i) is the sensitivity of alpha_i

if ~exist('p','var'), p = 8; end
N_alpha = r;
N_bin   = N_alpha * p;

alpha_center = alpha_tik;

% Per-variable spread: combine magnitude + uncertainty from inverse SV
res_norm   = norm(B_aug - A_aug * xi_tik);
sigma_alpha = 1 ./ s_diag(1:r);  % sensitivity per reduced coordinate
margin     = 3.0;
spread     = margin * (abs(alpha_center) + sigma_alpha .* res_norm);
R_enc_vec  = max(spread, 1e-6);  % floor to avoid zero range

% Build per-variable decoding: D_dec (N_alpha x N_bin)
step_sizes   = 2 * R_enc_vec / (2^p - 1);     % (r x 1)
alpha_offset = alpha_center - R_enc_vec;        % (r x 1)

D_dec = zeros(N_alpha, N_bin);
for i = 1:N_alpha
    bw_i = step_sizes(i) * (2.^(0:p-1));       % 1 x p, LSB first
    D_dec(i, (i-1)*p+1 : i*p) = bw_i;
end

fprintf('\n=== BINARY ENCODING ===\n');
fprintf('  p (bits/weight)   : %d\n', p);
fprintf('  N_alpha (reduced) : %d\n', N_alpha);
fprintf('  N_binary (qubits) : %d  (%d x %d)\n', N_bin, N_alpha, p);
fprintf('  median step size  : %.4e\n', median(step_sizes));
fprintf('  min/max R_enc     : [%.4e, %.4e]\n', min(R_enc_vec), max(R_enc_vec));


%% STEP 4 — BUILD QUBO

% E(q) = q'*Q*q + l'*q + const
% with Phi = A_red * D_dec,  B_tilde = B_aug - A_red * alpha_offset

fprintf('\nBuilding QUBO (%d x %d) ...\n', N_bin, N_bin);
t_build = tic;

Phi     = A_red * D_dec;
B_tilde = B_aug - A_red * alpha_offset;

Q_qubo     = Phi' * Phi;
l_qubo     = -2 * (Phi' * B_tilde);
const_qubo = norm(B_tilde)^2;

elapsed_build = toc(t_build);
fprintf('  build time : %.3f s\n', elapsed_build);
fprintf('  Q size     : %d x %d  (%.1f MB)\n', N_bin, N_bin, N_bin^2*8/1e6);


%% STEP 5 — SAVE QUBO DATA D-WAVE

fname = sprintf('qubo_v2_data_L%d_p%d.mat', L, p);

% Save physical-time vector for trajectory reconstruction
t_phys = t_train / w;

save(fname, ...
     'Q_qubo', 'l_qubo', 'const_qubo', ...
     'alpha_offset', 'step_sizes', 'R_enc_vec', 'alpha_center', ...
     'V_r', 'xi_tik', 'alpha_tik', ...
     'A_sys', 'B_sys', ...
     'p', 'N_bin', 'N_alpha', 'N_vars', 'r', 'L', 'n', ...
     'lambda_tik', 'tau_svd', ...
     'F', 'Fd', 'C1', 'C1d', 'C2', 'C2d', 'C3', 'C3d', 'H', ...
     'r_csp', 'w', 't_phys');

fprintf('  Saved to %s\n', fname);


%% STEP 6 — quantize alpha_tik to p bits
% Validates that the encoding can represent the Tikhonov solution.

alpha_int = round((alpha_tik - alpha_offset) ./ step_sizes);
alpha_int = max(0, min(2^p - 1, alpha_int));

q_warm = zeros(N_bin, 1);
for i = 1:N_alpha
    val = alpha_int(i);
    for k = 0:p-1
        q_warm((i-1)*p + k + 1) = floor(mod(val, 2^(k+1)) / 2^k);
    end
end

alpha_warm = alpha_offset + D_dec * q_warm;
xi_warm    = V_r * alpha_warm;
loss_warm  = A_sys * xi_warm - B_sys;

fprintf('\n=== WARM START (quantize Tikhonov to %d bits) ===\n', p);
fprintf('  max quantisation error  : %.4e  (max |alpha_warm - alpha_tik|)\n', max(abs(alpha_warm - alpha_tik)));
fprintf('  max |A*xi_warm - B|     : %.4e\n', max(abs(loss_warm)));
fprintf('  ||A*xi_warm - B||^2     : %.4e\n', norm(loss_warm)^2);


%% STEP 7 — SA FROM WARM START  (local refinement)

N_iter_warm = 2e5;
rng(42)

q_saw   = q_warm;
g_saw   = 2*Q_qubo*q_saw + l_qubo;
E_saw   = q_saw'*Q_qubo*q_saw + l_qubo'*q_saw;
q_bsaw  = q_saw;
E_bsaw  = E_saw;

dE_samp_w = zeros(200,1);
for s = 1:200
    k = randi(N_bin);
    delta = 1 - 2*q_saw(k);
    dE_samp_w(s) = abs(delta * g_saw(k) + Q_qubo(k,k));
end
T0_w   = mean(dE_samp_w) / log(5);
Tf_w   = T0_w * 1e-6;
cool_w = (Tf_w/T0_w)^(1/N_iter_warm);

fprintf('\n=== SA FROM WARM START (%d iterations, %d binary vars) ===\n', N_iter_warm, N_bin);
time_saw = tic;
T_w = T0_w;

for iter = 1:N_iter_warm
    k     = randi(N_bin);
    delta = 1 - 2*q_saw(k);
    dE    = delta * g_saw(k) + Q_qubo(k,k);

    if dE < 0 || rand() < exp(-dE/T_w)
        q_saw(k) = 1 - q_saw(k);
        g_saw    = g_saw + 2*delta*Q_qubo(:,k);
        E_saw    = E_saw + dE;
        if E_saw < E_bsaw
            E_bsaw = E_saw;
            q_bsaw = q_saw;
        end
    end
    T_w = T_w * cool_w;
end

elapsed_saw = toc(time_saw);
alpha_saw   = alpha_offset + D_dec * q_bsaw;
xi_saw      = V_r * alpha_saw;
loss_saw    = A_sys * xi_saw - B_sys;
fprintf('  elapsed : %.2f s\n', elapsed_saw);
fprintf('  max |A*xi_saw - B| : %.4e\n', max(abs(loss_saw)));


%% STEP 8 — SA FROM RANDOM START 

N_iter_cold = 5e5;
rng(123)

q_sac  = randi([0,1], N_bin, 1);
g_sac  = 2*Q_qubo*q_sac + l_qubo;
E_sac  = q_sac'*Q_qubo*q_sac + l_qubo'*q_sac;
q_bsac = q_sac;
E_bsac = E_sac;

dE_samp_c = zeros(500,1);
for s = 1:500
    k = randi(N_bin);
    delta = 1 - 2*q_sac(k);
    dE_samp_c(s) = abs(delta * g_sac(k) + Q_qubo(k,k));
end
T0_c   = mean(dE_samp_c) / log(5);
Tf_c   = T0_c * 1e-6;
cool_c = (Tf_c/T0_c)^(1/N_iter_cold);

fprintf('\n=== SA COLD START (%d iterations, %d binary vars) ===\n', N_iter_cold, N_bin);
time_sac = tic;
T_c = T0_c;

for iter = 1:N_iter_cold
    k     = randi(N_bin);
    delta = 1 - 2*q_sac(k);
    dE    = delta * g_sac(k) + Q_qubo(k,k);

    if dE < 0 || rand() < exp(-dE/T_c)
        q_sac(k) = 1 - q_sac(k);
        g_sac    = g_sac + 2*delta*Q_qubo(:,k);
        E_sac    = E_sac + dE;
        if E_sac < E_bsac
            E_bsac = E_sac;
            q_bsac = q_sac;
        end
    end
    T_c = T_c * cool_c;
end

elapsed_sac = toc(time_sac);
alpha_sac   = alpha_offset + D_dec * q_bsac;
xi_sac      = V_r * alpha_sac;
loss_sac    = A_sys * xi_sac - B_sys;
fprintf('  elapsed : %.2f s\n', elapsed_sac);
fprintf('  max |A*xi_cold - B| : %.4e\n', max(abs(loss_sac)));

%% STEP 9 — DIAGNOSTIC: lsqminnorm REFERENCE

time_lsq = tic;
xi_ref   = lsqminnorm(A_sys, B_sys);
elapsed_lsq = toc(time_lsq);
loss_ref = A_sys * xi_ref - B_sys;

%% STEP 10 — COMPARISON TABLE

fprintf('\n');
fprintf('%-35s  %12s  %12s  %12s  %12s  %12s\n', ...
    '', 'lsqminnorm', 'Tikhonov', 'warm start', 'SA (warm)', 'SA (cold)');
fprintf('%s\n', repmat('-',1,100));

rows = {
    'max |A*xi - B|',    max(abs(loss_ref)),  max(abs(loss_tik)),  max(abs(loss_warm)),  max(abs(loss_saw)),  max(abs(loss_sac));
    '||A*xi - B||^2',    norm(loss_ref)^2,    norm(loss_tik)^2,    norm(loss_warm)^2,    norm(loss_saw)^2,    norm(loss_sac)^2;
    '||xi||',            norm(xi_ref),         norm(xi_tik),        norm(V_r*alpha_warm), norm(xi_saw),        norm(xi_sac);
};

for row = 1:size(rows,1)
    label = rows{row,1};
    vals  = [rows{row,2}, rows{row,3}, rows{row,4}, rows{row,5}, rows{row,6}];
    fprintf('%-35s  %12.4e  %12.4e  %12.4e  %12.4e  %12.4e\n', label, vals);
end

%% STEP 11 — RECONSTRUCT TRAJECTORIES

t_phys = t_train / w;

[r1_ref,r2_ref,r3_ref,~,~,~,cv1_ref,cv2_ref,cv3_ref,J_ref] = ...
    reconstruct_traj(xi_ref, F,Fd,C1,C1d,C2,C2d,C3,C3d,H,L,r_csp,w,t_phys);

[r1_tik,r2_tik,r3_tik,~,~,~,cv1_tik,cv2_tik,cv3_tik,J_tik] = ...
    reconstruct_traj(xi_tik, F,Fd,C1,C1d,C2,C2d,C3,C3d,H,L,r_csp,w,t_phys);

[r1_warm,r2_warm,r3_warm,~,~,~,cv1_warm,cv2_warm,cv3_warm,J_warm] = ...
    reconstruct_traj(V_r*alpha_warm, F,Fd,C1,C1d,C2,C2d,C3,C3d,H,L,r_csp,w,t_phys);

[r1_saw,r2_saw,r3_saw,~,~,~,cv1_saw,cv2_saw,cv3_saw,J_saw] = ...
    reconstruct_traj(xi_saw, F,Fd,C1,C1d,C2,C2d,C3,C3d,H,L,r_csp,w,t_phys);

[r1_sac,r2_sac,r3_sac,~,~,~,cv1_sac,cv2_sac,cv3_sac,J_sac] = ...
    reconstruct_traj(xi_sac, F,Fd,C1,C1d,C2,C2d,C3,C3d,H,L,r_csp,w,t_phys);

fprintf('\n%-35s  %12s  %12s  %12s  %12s  %12s\n', ...
    '', 'lsqminnorm', 'Tikhonov', 'warm start', 'SA (warm)', 'SA (cold)');
fprintf('%-35s  %12.6f  %12.6f  %12.6f  %12.6f  %12.6f\n', ...
    'Cost J', J_ref, J_tik, J_warm, J_saw, J_sac);

%% STEP 12 — FIGURES

lbl = {'lsqminnorm','Tikhonov','warm start','SA (warm)','SA (cold)'};
col = {'b-', 'k-', 'g--', 'm--', 'r:'};

% Fig 1: Control comparison (lsqminnorm vs SA cold)
figure(1)
set(gcf,'Name','Control: Classical vs QUBO')
cv_ref_c  = {-cv1_ref, -cv2_ref, -cv3_ref};
cv_cold_c = {-cv1_sac, -cv2_sac, -cv3_sac};
for k = 1:3
    subplot(3,1,k); hold on; grid on
    plot(t_phys, cv_ref_c{k},  'b-', 'LineWidth',1.5, 'DisplayName','lsqminnorm')
    plot(t_phys, cv_cold_c{k}, 'r:', 'LineWidth',1.5, 'DisplayName','SA (QUBO)')
    ylabel(sprintf('u_%d', k))
    if k==1, legend('Location','best'); title('Control u = -\lambda_v  (QUBO v2)','FontWeight','Normal'); end
    if k==3, xlabel('t [s]'); end
end

% Fig 2: Position comparison (lsqminnorm vs SA cold)
figure(2)
set(gcf,'Name','Position: Classical vs QUBO')
r_ref_all  = {r1_ref, r2_ref, r3_ref};
r_cold_all = {r1_sac, r2_sac, r3_sac};
for k = 1:3
    subplot(3,1,k); hold on; grid on
    plot(t_phys, r_ref_all{k},  'b-', 'LineWidth',1.5, 'DisplayName','lsqminnorm')
    plot(t_phys, r_cold_all{k}, 'r:', 'LineWidth',1.5, 'DisplayName','SA (QUBO)')
    ylabel(sprintf('r_%d [m]', k))
    if k==1, legend('Location','best'); title('Position (QUBO v2)','FontWeight','Normal'); end
    if k==3, xlabel('t [s]'); end
end

% Fig 3: ODE residuals (lsqminnorm vs SA cold)
figure(3)
set(gcf,'Name','ODE Residuals')
hold on; grid on
plot(abs(loss_ref), 'b-', 'LineWidth',1.2, 'DisplayName','lsqminnorm')
plot(abs(loss_sac), 'r:', 'LineWidth',1.2, 'DisplayName','SA (QUBO)')
set(gca,'YScale','log')
xlabel('equation index'); ylabel('|A\xi - B|')
title('ODE residuals (QUBO v2)','FontWeight','Normal'); legend('Location','best')

% Fig 4: 3D trajectory (lsqminnorm vs SA cold)
figure(4)
set(gcf,'Name','3D Trajectory')
hold on; grid on; box on
plot3(r1_ref, r2_ref, r3_ref, 'b-', 'LineWidth',2.0, 'DisplayName','lsqminnorm')
plot3(r1_sac, r2_sac, r3_sac, 'r:', 'LineWidth',1.5, 'DisplayName','SA (QUBO)')
plot3(r1_ref(1),  r2_ref(1),  r3_ref(1),  'ko', ...
      'MarkerSize',10, 'MarkerFaceColor','g', 'DisplayName','start')
plot3(r1_ref(end), r2_ref(end), r3_ref(end), 'k^', ...
      'MarkerSize',10, 'MarkerFaceColor','r', 'DisplayName','target')
xlabel('r_1 [m]'); ylabel('r_2 [m]'); zlabel('r_3 [m]')
title('3D Trajectory (QUBO v2)','FontWeight','Normal')
legend('Location','best')
view(3)

%% QUBO DIAGNOSTICS

fprintf('\n=== QUBO DIAGNOSTICS (v2) ===\n');
fprintf('  Q size              : %d x %d\n', N_bin, N_bin);
fprintf('  Q symmetry err      : %.3e\n', norm(Q_qubo - Q_qubo','fro'));
fprintf('  Q sparsity          : %.2f%% nonzero\n', 100*nnz(Q_qubo > 0)/N_bin^2);
fprintf('  qubit count         : %d  (%d x %d)\n', N_bin, N_alpha, p);
fprintf('  vs v1 qubit count   : %d  (9 x %d x %d)\n', 9*L*p, L, p);
fprintf('  qubit savings       : %.1fx\n', (9*L*p) / N_bin);

fprintf('\nImprovement summary:\n');
fprintf('  [1] Tikhonov lambda = %.4e (unique, bounded solution)\n', lambda_tik);
fprintf('  [2] SVD rank %d -> %d fewer variables (%.1fx)\n', r, N_vars-r, N_vars/r);
fprintf('  [3] Per-variable ranges: median step = %.4e\n', median(step_sizes));
fprintf('  Warm start residual  : %.4e\n', norm(loss_warm)^2);
fprintf('  SA (warm) residual   : %.4e\n', norm(loss_saw)^2);
fprintf('  SA (cold) residual   : %.4e\n', norm(loss_sac)^2);

fprintf('\nTo solve on D-Wave:\n');
fprintf('  python solve_qubo_v2_dwave.py\n');
fprintf('  Variables: Q_qubo (%dx%d), l_qubo (%dx1)\n', N_bin, N_bin, N_bin);
fprintf('  Decode:  alpha = alpha_offset + D_dec * q\n');
fprintf('           xi    = V_r * alpha\n');

%% LOCAL FUNCTION

function [r1,r2,r3,v1,v2,v3,cv1,cv2,cv3,J] = reconstruct_traj(xi, F,Fd, ...
        C1,C1d, C2,C2d, C3,C3d, H, L, r_csp, w, t_phys)
    xis1  = xi(1:L);       xis2  = xi(L+1:2*L);   xis3  = xi(2*L+1:3*L);
    xicv1 = xi(6*L+1:7*L); xicv2 = xi(7*L+1:8*L); xicv3 = xi(8*L+1:9*L);

    r1  = (F*xis1  + C1 ) * r_csp;
    r2  = (F*xis2  + C2 ) * r_csp;
    r3  = (F*xis3  + C3 ) * r_csp;
    v1  = (Fd*xis1 + C1d) * (r_csp*w);
    v2  = (Fd*xis2 + C2d) * (r_csp*w);
    v3  = (Fd*xis3 + C3d) * (r_csp*w);
    cv1 = (H*xicv1) * (r_csp*w^2);
    cv2 = (H*xicv2) * (r_csp*w^2);
    cv3 = (H*xicv3) * (r_csp*w^2);
    J   = 0.5 * trapz(t_phys, cv1.^2 + cv2.^2 + cv3.^2);
end
