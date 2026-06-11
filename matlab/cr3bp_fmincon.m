function cr3bp_fmincon(params_path, results_path)
%CR3BP_FMINCON  Energy-optimal planar CR3BP transfer via Hermite-Simpson + fmincon.
%
%   cr3bp_fmincon(PARAMS_PATH, RESULTS_PATH) reads a JSON problem
%   specification from PARAMS_PATH, solves the energy-optimal low-thrust
%   CR3BP transfer with the *same* Hermite-Simpson direct transcription as
%   qalunar.reference.direct_collocation (which uses scipy SLSQP), but with
%   MATLAB's fmincon as the NLP solver, and writes the solution to
%   RESULTS_PATH as JSON.
%
%   This is an independent, cross-language validation of the Python
%   continuous baseline: identical problem and transcription, different
%   optimizer. Agreement in objective and trajectory confirms the baseline.
%
%   PARAMS JSON fields:
%       mu            scalar CR3BP mass parameter
%       r0, v0        1x2 initial position / velocity (synodic)
%       rf, vf        1x2 final position / velocity
%       T             time of flight (> 0)
%       n_intervals   number of Hermite-Simpson intervals
%       maxiter       fmincon MaxIterations
%       tol           optimality tolerance
%       control_bound symmetric box bound on |u_i|, or null for unbounded
%       algorithm     'sqp' or 'interior-point' (optional, default 'sqp')
%
%   Objective:  J = (1/2) * integral ||u||^2 dt   (trapezoidal)

p = jsondecode(fileread(params_path));

mu  = p.mu;
s0  = [p.r0(:)', p.v0(:)'];          % 1x4 initial state
sf  = [p.rf(:)', p.vf(:)'];          % 1x4 final state
T   = p.T;
N   = double(p.n_intervals);
nn  = N + 1;                          % number of nodes
h   = T / N;
tgrid = linspace(0, T, nn)';

if isfield(p, 'algorithm') && ~isempty(p.algorithm)
    algo = p.algorithm;
else
    algo = 'sqp';
end

% Trapezoidal quadrature weights [0.5, 1, ..., 1, 0.5].
trap = ones(nn, 1);
trap(1) = 0.5;
trap(end) = 0.5;

% ---- Initial guess ----
% If params carry an explicit packed seed ``z0`` (state.ravel followed by
% control.ravel, matching direct_collocation._pack), use it verbatim so the
% Python SLSQP and MATLAB fmincon multistarts share identical seeds.
% Otherwise fall back to the linear-interpolation seed with zero control.
if isfield(p, 'z0') && ~isempty(p.z0)
    z0 = p.z0(:);                          % column, length 6*nn
    assert(numel(z0) == 6*nn, 'z0 length %d != 6*(N+1)=%d', numel(z0), 6*nn);
else
    alphas = linspace(0, 1, nn)';
    S0 = (1 - alphas) .* s0 + alphas .* sf;   % nn x 4
    U0 = zeros(nn, 2);
    z0 = [S0(:); U0(:)];                       % column, length 6*nn
end

% ---- Optional control box bounds ----
if isfield(p, 'control_bound') && ~isempty(p.control_bound)
    ub_u = p.control_bound;
    lb = [-inf(nn*4, 1); -ub_u * ones(nn*2, 1)];
    ub = [ inf(nn*4, 1);  ub_u * ones(nn*2, 1)];
else
    lb = [];
    ub = [];
end

opts = optimoptions('fmincon', ...
    'Algorithm', algo, ...
    'SpecifyObjectiveGradient', true, ...
    'MaxIterations', double(p.maxiter), ...
    'MaxFunctionEvaluations', 1e7, ...
    'OptimalityTolerance', p.tol, ...
    'ConstraintTolerance', 1e-10, ...
    'StepTolerance', 1e-12, ...
    'Display', 'off');

obj     = @(z) objective(z, nn, h, trap);
nonlcon = @(z) constraints(z, nn, h, mu, s0, sf);

tic;
[zopt, fval, exitflag, output] = fmincon(obj, z0, [], [], [], [], lb, ub, nonlcon, opts);
solve_time = toc;

[S, U] = unpack(zopt, nn);
[c, ceq] = constraints(zopt, nn, h, mu, s0, sf); %#ok<ASGLU>
ndef = 4 * N;
defects = ceq(1:ndef);
bc_err  = ceq(ndef+1:end);

res = struct();
res.objective     = fval;
res.success       = exitflag > 0;
res.exitflag      = exitflag;
res.n_iterations  = output.iterations;
res.max_defect    = max(abs(defects));
res.max_bc_error  = max(abs(bc_err));
res.solve_time    = solve_time;
res.algorithm     = algo;
res.t  = tgrid(:)';
res.x  = S(:,1)';
res.y  = S(:,2)';
res.vx = S(:,3)';
res.vy = S(:,4)';
res.ux = U(:,1)';
res.uy = U(:,2)';

fid = fopen(results_path, 'w');
fwrite(fid, jsonencode(res), 'char');
fclose(fid);
end


% ======================================================================
function [S, U] = unpack(z, nn)
S = reshape(z(1:nn*4), nn, 4);
U = reshape(z(nn*4+1:end), nn, 2);
end


function f = rhs_batch(S, U, mu)
% Vectorized planar CR3BP RHS, identical to direct_collocation._rhs_batch.
x = S(:,1); y = S(:,2); vx = S(:,3); vy = S(:,4);
omu = 1 - mu;
dx1 = x + mu;
dx2 = x - omu;
r1_3 = (dx1.^2 + y.^2).^1.5;
r2_3 = (dx2.^2 + y.^2).^1.5;
ox = x - omu .* dx1 ./ r1_3 - mu .* dx2 ./ r2_3;
oy = y - omu .* y   ./ r1_3 - mu .* y   ./ r2_3;
ax = 2*vy + ox + U(:,1);
ay = -2*vx + oy + U(:,2);
f = [vx, vy, ax, ay];
end


function [J, g] = objective(z, nn, h, trap)
% J = (h/2) * sum_k trap_k * ||u_k||^2 ; analytic gradient supplied.
[~, U] = unpack(z, nn);
usq = U(:,1).^2 + U(:,2).^2;
J = 0.5 * h * sum(trap .* usq);
if nargout > 1
    g = zeros(numel(z), 1);
    gU = h * (trap .* U);          % nn x 2
    g(nn*4+1:end) = gU(:);
end
end


function [c, ceq] = constraints(z, nn, h, mu, s0, sf)
% Hermite-Simpson defects (4N) plus boundary conditions (8).
[S, U] = unpack(z, nn);
F = rhs_batch(S, U, mu);

Sk = S(1:end-1, :);  Skp1 = S(2:end, :);
Uk = U(1:end-1, :);  Ukp1 = U(2:end, :);
Fk = F(1:end-1, :);  Fkp1 = F(2:end, :);

Xmid = 0.5*(Sk + Skp1) + (h/8)*(Fk - Fkp1);
Umid = 0.5*(Uk + Ukp1);
Fmid = rhs_batch(Xmid, Umid, mu);

defect = Skp1 - Sk - (h/6)*(Fk + 4*Fmid + Fkp1);   % N x 4
bc = [S(1,:) - s0, S(end,:) - sf];                 % 1 x 8

c = [];
ceq = [defect(:); bc(:)];
end
