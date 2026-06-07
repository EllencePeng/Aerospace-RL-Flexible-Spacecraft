'''
Flexible spacecraft attitude tracking environment based on Xiao Yan's thesis.

Oct.21st 2025
Kinematics changed to MRP; removed omega_r terms.

Dec.9th 2025
Added fault injection and fault-tolerant control.

copyright @Joey
'''

import numpy as np
from scipy.integrate import solve_ivp

# ---------------- FlexibleSpacecraft class ----------------
class FlexibleSpacecraft():
    def __init__(self, env_time, dt, kp=30, kd=60, use_eta_reward=True,
                 use_param_uncertainty=False,
                 use_measurement_noise=False,
                 use_disturbance=False):
        '''Spacecraft structural parameters (nominal)'''
        self.J = np.array([[120., 3., 4.],
                    [3., 100., 10.],
                    [4., 10., 120.]])  # kg*m^2
        #self.J = np.diag([10.0, 15.0, 20.0])
        self.J_inv = np.linalg.inv(self.J)

        self.delta = np.array([
            [ 6.45637,  1.27814,  2.15629],
            [-1.25819,  0.91756, -1.67264],
            [ 1.11687,  2.48901, -0.83674],
            [ 1.23637, -2.65810, -1.12530]
        ])  # shape (4,3)

        # modal natural freqs and damping ratios
        self.omega_n = np.array([0.7681, 1.1038, 1.8733, 2.5496])  # rad/s (4 modes)
        self.xi = np.array([0.0056, 0.0086, 0.0128, 0.0252])

        '''Spacecraft initial state parameters'''
        self.kp = kp
        self.kd = kd
        self.use_eta_reward = use_eta_reward
        self.umax = 1

        # === Three uncertainty/disturbance switches ===
        self.use_param_uncertainty = use_param_uncertainty
        self.use_measurement_noise = use_measurement_noise
        self.use_disturbance = use_disturbance


        # Dynamics constant-term computation
        self.J_mb = self.J - self.delta.T.dot(self.delta)      # J_mb = J - delta^T delta  (delta is (m x 3), so delta.T@delta is 3x3)
        self.K = np.diag(self.omega_n**2)            # K = diag{omega_n^2}
        self.C = np.diag(2.0 * self.xi * self.omega_n)    # C = diag{2*xi*omega_n}
        m = self.K.shape[0]    # Number of modes included, 4 in this case
        self.A = np.block([[np.zeros((m, m)), np.eye(m)],
                           [-self.K, -self.C]])
        self.B = np.vstack([np.zeros((m, m)), np.eye(m)])
        self.m = m

        # === Store nominal parameters (set at __init__, never change) ===
        self.J_nom = self.J.copy()
        self.delta_nom = self.delta.copy()
        self.omega_n_nom = self.omega_n.copy()
        self.xi_nom = self.xi.copy()
        self.K_nom = self.K.copy()
        self.C_nom = self.C.copy()
        self.J_mb_nom = self.J_mb.copy()
        self.J_inv_nom = self.J_inv.copy()
        self.A_nom = self.A.copy()
        self.Jbar_nom = self.J_mb.copy()

        # === True dynamics parameters (point to nominal by default, replaced in reset when use_param_uncertainty) ===
        self.J_true = self.J
        self.delta_true = self.delta
        self.omega_n_true = self.omega_n
        self.xi_true = self.xi
        self.K_true = self.K
        self.C_true = self.C
        self.J_mb_true = self.J_mb
        self.A_true = self.A

        # === Measurement noise parameters ===
        self.noise_std_p = 8.0e-5       
        self.noise_std_omega = 5e-5   
        
        # === External disturbance parameters ===
        self.disturbance_type = 'none'  # 'none' | 'sin' | 'pulse' | 'continuous' | 'Hu'
        self.amplitudes = np.array([0.02, 0.03, 0.05])
        self.phases = np.array([1.2, 4.5, 0.9])
        self.omega_d = 0.1                   # ω_d for multi_sin disturbance (rad/s)
        
        # === Fault model ===
        self.fault_func = self.get_actuator_faults_0

        self.max_speed = 0.1
        self.dt = dt                      # Get control every dt seconds, integrate for dt

        '''Reinforcement learning parameters'''
        self.time_prev = 0
        self.i = 0      # Step counter; i total steps of dt each
        self.n = env_time/self.dt

        '''Observer parameters'''
        self.Jbar = self.J_mb
        self.N_modes = self.m
        self.theta = 5.0
        self.theta1 = 2.0
        self.theta2 = 5.0

        '''Robust control parameters'''
        # basic ISMC parameters
        self.Gamma = 2.0 * np.eye(3)       # sliding surface matrix Γ
        self.varepsilon = 1.0              # margin term ε in α upper bound (Eq. after Lemma 2)
        self.em = 0.5                      # actuator effectiveness loss upper bound e_m
        self.ubar_m = 1.0                  # additive fault bound ū_m (Assumption 3)
        self.d_m = 0.2                     # disturbance bound d_m (Assumption 1)
        # boundary layer width
        self.phi = 0.05

        # integral for s(t)
        self.int_term = np.zeros(3)

        # nominal control parameters
        self.p2 = 0.2

        # # Adaptive FT parameters
        # self.W_hat = 0.0
        # self.gamma = 1.0          # Adaptive gain (adjusts estimation rate)
        # self.epsilon_a = 1e-3       # Small margin term, used to avoid division by zero

        # Adaptive FT parameters (Section 5.1, Eqs. 15-16)
        self.alpha_hat = 1.0             # adaptive switching gain α̂
        self.lam = 0.00025               # leakage term λ in adaptive law (Eq. 16)
        self.gamma_aft = 10.0            # adaptive gain rate γ in adaptive law (Eq. 16)
        self.epsilon = 0.01              # boundary layer thickness ϵ (Eq. 15)


    def _recompute_true_params(self):
        """Recompute derived parameters from true δ, ω_n, ξ, J"""
        self.K_true = np.diag(self.omega_n_true**2)
        self.C_true = np.diag(2.0 * self.xi_true * self.omega_n_true)
        self.J_mb_true = self.J_true - self.delta_true.T @ self.delta_true
        self.A_true = np.block([
            [np.zeros((self.m, self.m)), np.eye(self.m)],
            [-self.K_true, -self.C_true]
        ])

    def _get_disturbance(self, t):
        if self.disturbance_type == 'sin':
            return self.d_disturbance_sin(t)
        elif self.disturbance_type == 'pulse':
            return self.d_disturbance_pulse(t)
        elif self.disturbance_type == 'continuous':
            return self.d_disturbance_continuous(t)
        elif self.disturbance_type == 'Hu':
            return self.d_disturbance_Hu(t)
        else:
            return np.zeros(3)

    def reset(self, mode= 'train'):
        self.mode = mode

        # Parameter uncertainty: default true = nominal
        self.J_true = self.J_nom.copy()
        self.delta_true = self.delta_nom.copy()
        self.omega_n_true = self.omega_n_nom.copy()
        self.xi_true = self.xi_nom.copy()
        self._recompute_true_params()

        #--------- Spacecraft model initialization parameters -----------------
        self.p0 = np.zeros(3)   # initial spacecraft quaternion q(t0)
        self.omega0 = np.zeros(3)    # initial body angular velocity
        eta0 = np.zeros(self.m)                       # initial modal coord (assumed zero)
        chi0 = np.zeros(self.m)                       # initial modal velocity (assumed zero)
        self.q_true = np.zeros(3)  # Initial q for observer computation

        if mode == 'Scenario1' or mode == 'test5':
            self.p0 = np.array([0.3,0.1,-0.2])

        elif mode == 'train':
            self.p0 = np.round(np.random.uniform(-1, 1, 3), decimals=1)
            self.disturbance_type = 'sin'
            self.amplitudes = np.random.uniform(0, 0.05, 3)
            self.phases = np.random.uniform(0, 2*np.pi, 3)

        elif mode == 'Scenario2':       # Nonzero initial angular velocity
            self.p0 = np.array([0.3,0.1,-0.2])
            self.omega0 = np.array([0.02,-0.03,0.04])

        elif mode == 'test2':       # Impulsive disturbance
            self.p0 = np.array([0.3,0.1,-0.2])
            self.disturbance_type = 'pulse'

        elif mode == 'test3':       # Continuous disturbance after 100s
            self.p0 = np.array([0.3,0.1,-0.2])
            self.disturbance_type = 'continuous'

        elif mode == 'Scenario0':       # Pure baseline: initial attitude only, no disturbance/noise/mismatch
            self.p0 = np.array([0.3,0.1,-0.2])

        elif mode == 'Scenario3':       # J mismatch + flexible parameter mismatch + external disturbance
            self.p0 = np.array([0.3,0.1,-0.2])
            # J mismatch
            self.J_true = np.array([[120., 6., 7.],
                                    [6., 100., 10.],
                                    [7., 10., 120.]])
            # Flexible parameter mismatch +10%
            self.delta_true = self.delta_nom * 1.10
            self.omega_n_true = self.omega_n_nom * 1.10
            self.xi_true = self.xi_nom * 1.10
            self.disturbance_type = 'Hu'
            self._recompute_true_params()

        elif mode == 'Scenario4':
            self.p0 = np.array([0.3,0.1,-0.2])
            self.fault_func = self.get_actuator_faults
            self.disturbance_type = 'Hu'
            # J mismatch
            self.J_true = np.array([[120., 6., 7.],
                                    [6., 100., 10.],
                                    [7., 10., 120.]])
            # Flexible parameter mismatch +10%
            self.delta_true = self.delta_nom * 1.10
            self.omega_n_true = self.omega_n_nom * 1.10
            self.xi_true = self.xi_nom * 1.10
            self._recompute_true_params()
        else: raise ValueError('Mode must be one of "train" "Scenario0" "Scenario1" "Scenario2" "Scenario3" "Scenario4" "test2" "test3" "test5"')

        # state vector x0 ordering: [q0, qv(3), omega_e(3), eta(m), chi(m)]
        self.state = np.hstack([self.p0, self.omega0, eta0, chi0])
        self.state_obs = np.hstack([self.p0, np.zeros_like(self.omega0), 
                                            np.zeros_like(eta0), 
                                            np.zeros_like(chi0)])
        
        # _v suffix: uses v = qdot as intermediate state variable ([q0, v0, eta0, psi0])
        # _obs suffix: uses observer-estimated values
        self.state_v = np.hstack([self.p0, self.omega0, eta0, chi0])
        self.state_obsv = np.hstack([self.p0, np.zeros_like(self.omega0), 
                                            np.zeros_like(eta0), 
                                            np.zeros_like(chi0)])
        self.state_out = np.hstack([self.p0, self.omega0])

        # Sensor output (equals true state when no noise)
        self.p_sensor = self.p0.copy()
        self.omega_sensor = self.omega0.copy()
        
        #--------- RL environment initialization parameters ---------------
        self.p_norm_prev = np.linalg.norm(self.state[0:3], ord=1)
        self.omega_norm_prev = np.linalg.norm(self.state[3:6], ord=1)
        self.time_prev = 0
        
        #--------- Maintained parameters ----------------
        self.X_sol = self.state.reshape(14, 1)          # X: true dynamics simulation
        self.Xobs_sol = self.state_obs.reshape(14, 1)   # Xobs: observer simulation

        self.Xv_sol = self.state_v.reshape(14, 1)       # _v suffix: uses v = qdot as intermediate state
        self.Xobsv_sol = self.state_obsv.reshape(14, 1)

        self.Xout_sol = self.state_out.reshape(6, 1)  # True attitude q, observer states for rest
        
        self.t_sol = np.array([self.time_prev])
        self.Tr_sol = np.array([[0],[0],[0]])
        self.d_sol = np.array([[0],[0],[0]])
        self.i = 0
        self.t = 0
        
        
        #return self.state
        return self.state_out

    def step(self, Tr):  # Tr input is 3-D ndarray
        self.i += 1 
        dt = self.dt
        self.t = self.i * dt

        self.t_sol = np.append(self.t_sol, self.t)
        #self.Tr_sol = np.column_stack((self.Tr_sol, Tr)) # Moved outside
        
        # ------------------ Disturbance and faults ---------------------
        if self.use_disturbance:
            d = self._get_disturbance(self.t)
        else:
            d = np.zeros(3)
        self.d_sol = np.column_stack((self.d_sol, d))

        E, ubar = self.fault_func(self.t)

        # Actuator output (with faults and disturbance)
        Tr = (np.eye(3) - E) @ Tr + ubar + d
        # ------------------ Disturbance and faults end -----------------

        tspan = np.linspace(self.time_prev,self.t,100) # Subdivide each dt into 100 integration steps
        tspanivp = (tspan[0], tspan[-1])

        # --- True dynamics computation ----
        sol_x = solve_ivp(fun=lambda t, X: self.dynamics(t, X, Tr),
                        t_span=tspanivp,
                        y0=self.X_sol[:,-1],
                        t_eval=tspan,
                        method='RK45')

        self.state = sol_x.y[:,-1]
        self.q_true = self.state[0:3]  # True MRP for updating observer at this step
        self.omega_true = self.state[3:6]
        self.X_sol = np.column_stack((self.X_sol, self.state))

        # ============ Sensor output (unified measurement interface) ============
        if self.use_measurement_noise:
            self.p_sensor = self.q_true + np.random.normal(0, self.noise_std_p, 3)
            self.omega_sensor = self.omega_true + np.random.normal(0, self.noise_std_omega, 3)
        else:
            self.p_sensor = self.q_true
            self.omega_sensor = self.omega_true
        
        # # --- True dynamics (v-form) computation ----
        # sol_xv = solve_ivp(fun=lambda t, X: self.dynamics_v(t, X, Tr),
        #                 t_span=tspanivp,
        #                 y0=self.Xv_sol[:,-1],
        #                 t_eval=tspan,
        #                 method='RK45')
        
        # self.state_v = sol_xv.y[:,-1]
        # self.q_true = self.state_v[0:3]  # True MRP for updating observer at this step
        # self.omega_true = self.state[3:6]
        # self.Xv_sol = np.column_stack((self.Xv_sol, self.state_v))
        
        # --- Observer update ----
        sol_xobsv = solve_ivp(fun=lambda t, X: self.observer(t, X, Tr),
                            t_span=tspanivp,
                            y0=self.Xobsv_sol[:,-1],
                            t_eval=tspan,
                            method='RK45')
        self.state_obsv = sol_xobsv.y[:,-1]
        self.Xobsv_sol = np.column_stack((self.Xobsv_sol, self.state_obsv))
        
        # --- Maintain X / Xobs (omega form) ---
        # # ---- True system (omega form) ----
        # q = self.state_v[0:3]
        # v = self.state_v[3:6]
        # eta = self.state_v[6:6+self.m]
        # psi = self.state_v[6+self.m:6+2*self.m]

        # P = np.linalg.inv(self.T_matrix(q))
        # omega = P @ v

        # self.state = np.concatenate([q, omega, eta, psi])
        # self.X_sol = np.column_stack((self.X_sol, self.state))

        # ---- Observer (omega form) ----
        q_hat = self.state_obsv[0:3]
        v_hat = self.state_obsv[3:6]
        eta_hat = self.state_obsv[6:6+self.m]
        psi_hat = self.state_obsv[6+self.m:6+2*self.m]
        
        P_hat = np.linalg.inv(self.T_matrix(q_hat))
        omega_hat = P_hat @ v_hat

        self.state_obs = np.concatenate([q_hat, omega_hat, eta_hat, psi_hat])
        self.Xobs_sol = np.column_stack((self.Xobs_sol, self.state_obs))
        
        self.state_out = np.concatenate([self.p_sensor, self.omega_sensor])
        self.Xout_sol = np.column_stack((self.Xout_sol, self.state_out))

        self.time_prev = self.t
        
        # Extract states for reward computation
        p = self.state_out[0:3]
        omega = self.state_out[3:6]
        eta = eta_hat
        Tr_prev = self.Tr_sol[:,-2]
        
        p_norm = np.linalg.norm(p, ord=1)
        omega_norm = np.linalg.norm(omega, ord=1)
        eta_norm = np.linalg.norm(eta, ord=2)
        
        #===================== status define / ending criteria ======================#
        exceed = np.max(np.abs(p))>=1 or np.max(np.abs(omega))>=1
        done = self.i >= self.n or exceed
        
        #===================== Step Reward ======================#
        r = 0
        if not done:
            if p_norm >= 0.5:
                #------ Base reward -----#
                if p_norm < self.p_norm_prev:
                    r = 1 - np.sum(Tr**2)
                elif p_norm >= self.p_norm_prev and omega_norm < self.omega_norm_prev:
                    r = 1 - np.sum(Tr**2)
                else:
                    r = - p_norm - 10*omega_norm - np.sum(Tr**2)

            #------ Convergence reward -----
            if p_norm < 0.5:
                r += 1 / (p_norm+ 0.01)
                r += 0.05 / (omega_norm+ 0.01) 
                if self.use_eta_reward:
                    r += 0.05 / (eta_norm+ 0.01)
                
            #------ Smoothness / stability reward -----
            if p_norm < 0.01:
                r -= 10*np.sum(Tr**2)    # Penalize excessive torque variation
            r -= 10*np.linalg.norm(Tr - Tr_prev, 1)


            #------ Terminal reward -----
        else:
            if exceed: 
                r = -100
            elif p_norm < 0.01: 
                r = 1000
            elif p_norm < 0.05: 
                r = 200
            elif p_norm < 0.1: 
                r = 100
            else: 
                r = 0

        self.p_norm_prev = p_norm
        self.omega_norm_prev = omega_norm

        #===================== status define / ending criteria ======================#
        done = self.i >= self.n 

        # return self.state, r, done, False
        return self.state_out, r, done, False
    
    def get_actuator_faults(self, t):
        """
        Compute actuator fault parameters at time t.

        Args:
        t: time (float)

        Returns:
        E: actuator effectiveness matrix (3x3 diagonal)
        ubar: actuator bias vector (3-D vector)
        """
        if t < 10:
            E = np.diag([0, 0, 0])   # no fault
            ubar = np.zeros(3)
        elif t < 50:
            E = np.diag([0.5, 0.5, 0.5])  # partial loss
            ubar = np.zeros(3)
        else:
            E = np.diag([0.5, 0.5, 0.5])
            ubar = 0.3 + 0.05 * np.sin(t) * np.ones(3)
            
        return E, ubar
    #u = self.controller(q[1:4], Rqe, omega_e, omega_r_bar, omega_r_dot, eta, chi, d)
    # def controller(self):
    #     p = self.state[0:3]
    #     omega = self.state[3:6]
        
    #     u2 = -self.k_p * p - self.k_d * omega

    #     u = np.clip(u2, -self.umax, self.umax)
    #     return u
    
    @staticmethod
    def get_actuator_faults_0(t):
        """
        Default actuator fault function, returns no-fault parameters.

        Args:
        t: time (float)

        Returns:
        E: actuator effectiveness matrix (3x3 diagonal), default no fault
        ubar: actuator bias vector (3-D vector), default zero
        """
        E = np.diag([0, 0, 0])   # No fault
        ubar = np.zeros(3)       # Zero bias vector
        return E, ubar

    # ---------------------------------------------
    # Nominal controller (MRP-based PD + tanh)
    # ---------------------------------------------
    def nominal_control(self, p, omega):

        # saturated PD
        # u_nom = - self.kp * p - self.kd * np.tanh(omega / self.p2)
        u_nom = term1 = -self.kp * p - self.kd * omega
        u_nom = np.clip(u_nom, -self.umax, self.umax)
        return u_nom


    # ------------------------------------------------
    # Compute sliding surface (ISMC) — Equation (14)
    # ------------------------------------------------
    def compute_sliding_surface(self, omega, u_nom):

        # dynamics term:  -ω×Jω
        w_cross = np.array([
            [0, -omega[2], omega[1]],
            [omega[2], 0, -omega[0]],
            [-omega[1], omega[0], 0]
        ])
        gyro = - w_cross @ (self.J @ omega)

        # integrate  J⁻¹( -ω×Jω + u_nom )
        self.integral_term += self.J_inv @ (gyro + u_nom) * self.dt

        # sliding variable
        s = self.Gamma @ (omega - self.omega0 - self.integral_term)

        return s

    # ------------------------------------------------
    # Upper bound α of adaptive gain — Lemma 2
    # ------------------------------------------------
    def compute_alpha_bound(self, u_nom):
        norm_unom_inf = np.max(np.abs(u_nom))

        alpha_bound = ( np.sqrt(3) * self.em * norm_unom_inf + self.ubar_m + self.d_m + self.varepsilon ) \
              / (1.0 - self.em)

        return alpha_bound

    def update_alpha_hat(self):
        GammaJ_inv_T = (self.Gamma @ np.linalg.inv(self.J)).T
        GammaJ_inv_T_s_norm = np.linalg.norm(GammaJ_inv_T @ self.s)

        alpha_hat_dot = self.gamma_aft * (GammaJ_inv_T_s_norm - self.lam * self.alpha_hat)
        self.alpha_hat += alpha_hat_dot * self.dt

        self.alpha_hat = max(self.alpha_hat, 0.0)

    def adaptive_switching_control(self):
        GammaJ_inv_T = (self.Gamma @ np.linalg.inv(self.J)).T
        v = GammaJ_inv_T @ self.s
        v_norm = np.linalg.norm(v)

        if v_norm < 1e-8:
            return np.zeros(3)

        if self.alpha_hat * v_norm >= self.epsilon:
            u_aN = - self.alpha_hat * v / v_norm
        else:
            u_aN = - (self.alpha_hat ** 2 / self.epsilon) * v

        return u_aN

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    # -----------------------------
    # saturation function (vector)
    # -----------------------------
    def sat(self, z):
        z_sat = np.zeros_like(z)
        for i in range(3):
            if z[i] > 1:
                z_sat[i] = 1
            elif z[i] < -1:
                z_sat[i] = -1
            else:
                z_sat[i] = z[i]
        return z_sat
    # main control
    def controllerBFT(self):
        p = self.state[0:3]  # Attitude error
        omega = self.state[3:6]  # Angular velocity

        unom = self.nominal_control(p, omega)

        # integral term for sliding surface
        drift = -np.cross(omega, self.J @ omega) + unom
        self.int_term += self.J_inv @ drift * self.dt

        # sliding surface
        self.s = self.Gamma @ (omega - self.int_term)

        # compute direction vector v = (Γ J^{-1})^T s
        v = self.Gamma.T @ self.J_inv.T @ self.s

        # boundary layer smoothing: sat(v/phi)
        v_norm = v / self.phi
        v_sat = self.sat(v_norm)

        alpha_bound = self.compute_alpha_bound(unom)
        uN = -alpha_bound * v_sat
        u = np.clip(unom + uN, -self.umax, self.umax)
        return u
    
    def controllerTD3BFT(self,unom):
        p = self.state[0:3]  # Attitude error
        omega = self.state[3:6]  # Angular velocity

        # integral term for sliding surface
        drift = -np.cross(omega, self.J @ omega) + unom
        self.int_term += self.J_inv @ drift * self.dt

        # sliding surface
        s = self.Gamma @ (omega - self.int_term)

        # compute direction vector v = (Γ J^{-1})^T s
        v = self.Gamma.T @ self.J_inv.T @ s

        # boundary layer smoothing: sat(v/phi)
        v_norm = v / self.phi
        v_sat = self.sat(v_norm)

        alpha_bound = self.compute_alpha_bound(unom)
        uN = -alpha_bound * v_sat
        u = np.clip(unom + uN, -self.umax, self.umax)
        return u

    def controllerAFT(self):
        p = self.p_sensor      # Sensor-output MRP
        omega = self.omega_sensor  # Sensor-output angular velocity

        unom = self.nominal_control(p, omega)

        # Use observer to generate eta first/second derivatives
        current_observer_dot = self.observer(self.t, self.Xobsv_sol[:,-1], self.Tr_sol[:,-1])
        eta_dot = current_observer_dot[6:6+self.m]
        eta_dotdot = current_observer_dot[6+self.m:6+2*self.m]
        # integral term for sliding surface
        # drift = -np.cross(omega, self.J @ omega) + unom
        drift = -np.cross(omega, (self.J @ omega + self.delta.T @ eta_dot)) - self.delta.T @ eta_dotdot + unom  # flexible term added

        self.int_term += self.J_inv @ drift * self.dt

        # sliding surface (Eq. 14)
        self.s = self.Gamma @ (omega - self.int_term)

        # 3. Update adaptive gain
        self.update_alpha_hat()

        # 4. Adaptive switching control
        u_aN = self.adaptive_switching_control()

        u = np.clip(unom + u_aN, -self.umax, self.umax)
        return u

    def controllerTD3AFT(self,unom):
        p = self.p_sensor      # Sensor-output MRP
        omega = self.omega_sensor  # Sensor-output angular velocity

        # Use observer to generate eta first/second derivatives
        current_observer_dot = self.observer(self.t, self.Xobsv_sol[:,-1], self.Tr_sol[:,-1])
        eta_dot = current_observer_dot[6:6+self.m]
        eta_dotdot = current_observer_dot[6+self.m:6+2*self.m]
        # integral term for sliding surface
        #drift = -np.cross(omega, self.J @ omega) + unom
        drift = -np.cross(omega, (self.J @ omega + self.delta.T @ eta_dot)) - self.delta.T @ eta_dotdot + unom  # flexible term added

        self.int_term += self.J_inv @ drift * self.dt

        # sliding surface (Eq. 14)
        self.s = self.Gamma @ (omega - self.int_term)

        # 3. Update adaptive gain
        self.update_alpha_hat()

        # 4. Adaptive switching control
        u_aN = self.adaptive_switching_control()

        u = np.clip(unom + u_aN, -self.umax, self.umax)
        return u
 
    def dynamics(self, t, x, Tr):
        # x layout: [p(3), omega_e(3), eta(4), chi(4)]
        p = x[0:3]
        omega = x[3:6]
        eta = x[6:6+self.m]
        chi = x[6+self.m:6+2*self.m]

        # p derivative
        pdot = self.p_derivative(p, omega)

        # Select nominal or true dynamics parameters based on switch
        if self.use_param_uncertainty:
            J_mb, delta, K, C, A = self.J_mb_true, self.delta_true, self.K_true, self.C_true, self.A_true
        else:
            J_mb, delta, K, C, A = self.J_mb_nom, self.delta_nom, self.K_nom, self.C_nom, self.A_nom

        # Θ matrix
        Theta = self.hat(J_mb @ omega) + self.hat(delta.T @ chi)

        # RHS of 3-4
        RHS = Theta @ omega + delta.T @ (K @ eta + C @ chi - C @ delta @ omega) + Tr
        # solve for omega_dot
        try:
            omega_dot = np.linalg.solve(J_mb, RHS)
        except np.linalg.LinAlgError:
            omega_dot = np.linalg.pinv(J_mb).dot(RHS)

        # modal equations (3-5)
        big_state = np.hstack([eta, chi])
        eta_chi_dot = A @ big_state - A @ self.B @ delta @ omega

        # assemble derivative
        xdot = np.zeros_like(x)
        xdot[0:3] = pdot
        xdot[3:6] = omega_dot
        xdot[6:6+self.m] = eta_chi_dot[:self.m]
        xdot[6+self.m:6+2*self.m] = eta_chi_dot[self.m:]
        return xdot
    
    #------- Observer-related functions -------------
    def dynamics_v(self, t, X, Tr):
        q = X[0:3]
        v = X[3:6]
        eta = X[6:6 + self.N_modes]
        psi = X[6 + self.N_modes:6 + 2 * self.N_modes]

        T = self.T_matrix(q)
        P = np.linalg.inv(T)

        q_dot = v
        f = self.f_nonlinear(q, v, eta, psi)
        g = self.g_matrix(q)
        v_dot = f + g @ Tr

        eta_dot = psi - self.delta @ (P @ v)
        psi_dot = - self.C @ psi - self.K @ eta + self.C @ self.delta @ (P @ v)

        return np.concatenate([q_dot, v_dot, eta_dot, psi_dot])

    def observer(self, t, X_obs, Tr):
        q_hat = X_obs[0:3]
        v_hat = X_obs[3:6]
        eta_hat = X_obs[6:6 + self.N_modes]
        psi_hat = X_obs[6 + self.N_modes:6 + 2 * self.N_modes]

        q = self.p_sensor
        q_tilde = q - q_hat

        T_hat = self.T_matrix(q_hat)
        P_hat = np.linalg.inv(T_hat)

        f_hat = self.f_nonlinear(q, v_hat, eta_hat, psi_hat)
        g = self.g_matrix(q)

        q_hat_dot = v_hat + self.theta * self.theta1 * q_tilde
        v_hat_dot = f_hat + g @ Tr + self.theta**2 * self.theta2 * q_tilde

        eta_hat_dot = psi_hat - self.delta @ (P_hat @ v_hat)
        psi_hat_dot = - self.C @ psi_hat - self.K @ eta_hat + self.C @ self.delta @ (P_hat @ v_hat)

        return np.concatenate([q_hat_dot, v_hat_dot, eta_hat_dot, psi_hat_dot])

    def f_nonlinear(self, q, v, eta, psi):
        T = self.T_matrix(q)
        P = np.linalg.inv(T)
        T_dot = self.T_dot(q, v)
        P_dot = -P @ T_dot @ P
        omega = P @ v
        omega_cross = self.skew(omega)

        term1 = - T @ (P_dot @ v)
        term2 = - T @ np.linalg.inv(self.Jbar) @ (
            omega_cross @ (self.Jbar @ omega + self.delta.T @ psi)
        )
        term3 = T @ np.linalg.inv(self.Jbar) @ (
            self.delta.T @ (self.C @ psi + self.K @ eta - self.C @ self.delta @ omega)
        )
        return term1 + term2 + term3

    def g_matrix(self, q):
        return self.T_matrix(q) @ np.linalg.inv(self.Jbar)

    def T_matrix(self, q):
        q_cross = self.skew(q)
        q_norm2 = q @ q
        return 0.5 * ((1 - q_norm2 / 2) * np.eye(3) + q_cross + np.outer(q, q))

    def T_dot(self, q, v):
        return 0.5 * (-(q @ v) * np.eye(3) + self.skew(v) + 2 * np.outer(q, v))

    def skew(self, x):
        return np.array([[0, -x[2], x[1]],
                         [x[2], 0, -x[0]],
                         [-x[1], x[0], 0]])
    #------- Observer-related functions end ----------
    
    def d_disturbance_sin(self, t):
        # d(t) from image: small offset + sin terms (N*m)
        # return np.array([
        #     0.2 * np.sin(0.1*t + 0.25*np.pi) + 1e-7,
        #     -0.1 * np.sin(0.1*t - 0.5*np.pi) + 1e-7,
        #     0.3 * np.sin(0.1*t + 0.6*np.pi) + 1e-7
        # ])
        "Random phase/amplitude version below"
        # Unpack amplitude and phase parameters
        amplitude1, amplitude2, amplitude3 = self.amplitudes
        phase1, phase2, phase3 = self.phases
        
        # d(t) from image: small offset + sin terms (N*m)
        return np.array([
            amplitude1 * np.sin(2*np.pi/40*t + phase1) + 1e-7,
            amplitude2 * np.sin(2*np.pi/40*t + phase2) + 1e-7,
            amplitude3 * np.sin(2*np.pi/40*t + phase3) + 1e-7
        ])
    def simulate(self, x0, t_span, t_eval, method='RK45', max_step=0.1):
        sol = solve_ivp(self.dynamics, t_span, x0, t_eval=t_eval, method=method, max_step=max_step)
        # normalize quaternion at every saved time
        for i in range(sol.y.shape[1]):
            q = sol.y[0:4, i]
            n = np.linalg.norm(q)
            if n > 1e-12:
                sol.y[0:4, i] = q / n
        return sol
    def d_disturbance_Hu(self, t):
        w = self.omega_d
        return 0.2e-3 * np.array([
            3*np.cos(10*w*t) + 4*np.sin(3*w*t) - 10,
            -1.5*np.sin(2*w*t) + 3*np.cos(5*w*t) + 15,
            3*np.sin(10*w*t) - 8*np.sin(4*w*t) + 10
        ])

    @staticmethod
    def d_disturbance_0(t):
        return np.array([0,0,0])


    @staticmethod
    def d_disturbance_pulse(t):
        if t >= 60 and t <= 63:
            return np.array(np.array([2,-1.5,1]))
        return np.array([0,0,0])
    @staticmethod
    def d_disturbance_continuous(t):
        if t >= 100:
            return np.array([0.0,0.0,0.6])
        return np.array([0,0,0])
    # ---------------- helpers ----------------
    @staticmethod
    def hat(v):
        x, y, z = v
        return np.array([[0, -z, y],
                        [z, 0, -x],
                        [-y, x, 0]])

    # ---------------- Eq. 3-3 ----------------
    @staticmethod
    def quat_derivative(q, omega):
        q0 = q[0]
        qv = q[1:4]
        qv_hat = FlexibleSpacecraft.hat(qv)
        qv_dot = 0.5 * (qv_hat + q0*np.eye(3)).dot(omega)
        q0_dot = -0.5 * qv.dot(omega)
        return np.hstack([q0_dot, qv_dot])   # Horizontally stack scalar and vector into 4-D vector
    # ---------------- MRP kinematics formula ----------------
    @staticmethod
    def p_derivative(p, omega):
        p_cross = np.array([[0, -p[2], p[1]],
                            [p[2], 0, -p[0]],
                            [-p[1], p[0], 0]], dtype=float)
        
        Gp = (1- p.T@p)*np.eye(3) + 2*p_cross + 2*p.reshape(-1, 1) @ p.reshape(1, -1)
        dpdt = 1/4 * Gp @ omega
        return dpdt

    # ---------------- Expression R ----------------
    @staticmethod
    def R_transform(q):
        """
        Compute R = (q0^2 - qv^T qv) I + 2 qv qv^T - 2 q0 hat(qv)
        q: array-like, [q0, qx, qy, qz]
        Returns 3x3 direction cosine matrix.
        """
        q = np.asarray(q, dtype=float)
        q0 = q[0]
        qv = q[1:4]
        I3 = np.eye(3)
        term1 = (q0*q0 - np.dot(qv, qv)) * I3
        term2 = 2.0 * np.outer(qv, qv)
        term3 = -2.0 * qv @ FlexibleSpacecraft.hat(qv)
        return term1 + term2 + term3
    
    
    @staticmethod
    def calculate_error_simple(q, q_r, omega, omega_r):
        """
        Compute error between current and reference values.

        Args:
        q: current attitude quaternion [q0, qx, qy, qz]
        q_r: reference attitude quaternion [q0, qx, qy, qz]
        omega: current angular velocity [wx, wy, wz]
        omega_r: reference angular velocity [wx, wy, wz]

        Returns:
        qe: attitude error quaternion
        omega_e: angular velocity error vector
        """
        # Ensure inputs are numpy arrays
        q = np.array(q)
        q_r = np.array(q_r)
        omega = np.array(omega)
        omega_r = np.array(omega_r)
        
        # Separate scalar and vector parts of quaternion
        q0, qv = q[0], q[1:4]
        qr0, qrv = q_r[0], q_r[1:4]
        
        # Compute attitude error quaternion qe = q_r* ⊗ q
        # Formula: qe = [qr0*q0 + qrv^T*qv, qr0*qv - q0*qrv - qrv×qv]
        qe_scalar = qr0 * q0 + np.dot(qrv, qv)  # qr0*q0 + qrv^T*qv
        qe_vector = qr0 * qv - q0 * qrv - FlexibleSpacecraft.hat(qrv) @ qv  # qr0*qv - q0*qrv - qrv×qv
        
        qe = np.array([qe_scalar, qe_vector[0], qe_vector[1], qe_vector[2]])
        
        # Normalize
        qe_norm = np.linalg.norm(qe)
        if qe_norm < 1e-10:
            raise ValueError("Attitude error quaternion norm too small")
        qe = qe / qe_norm
        
        # Compute angular velocity error omega_e = omega - R(qe) * omega_r
        omega_e = omega - FlexibleSpacecraft.R_transform(qe) @ omega_r
        
        return qe, omega_e
    @staticmethod
    def random_quaternion():
        """Generate random unit quaternion"""
        # Method 1: uniform distribution on sphere
        u1, u2, u3 = np.random.random(3)
        
        # Uniform sampling using Hopf coordinates
        sqrt_u1 = np.sqrt(u1)
        sqrt_1_minus_u1 = np.sqrt(1 - u1)
        
        q0 = sqrt_1_minus_u1 * np.sin(2 * np.pi * u2)
        qx = sqrt_1_minus_u1 * np.cos(2 * np.pi * u2)
        qy = sqrt_u1 * np.sin(2 * np.pi * u3)
        qz = sqrt_u1 * np.cos(2 * np.pi * u3)
        
        return np.array([q0, qx, qy, qz])