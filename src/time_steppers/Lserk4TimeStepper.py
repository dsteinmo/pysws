class Lserk4TimeStepper:

    LSERK4_STAGES = range(0,5)
    
    RK4A = [            0.0,
        -567301805773.0/1357537059087.0,
        -2404267990393.0/2016746695238.0,
        -3550918686646.0/2091501179385.0,
        -1275806237668.0/842570457699.0];

    RK4B = [ 1432997174477.0/9575080441755.0,
         5161836677717.0/13612068292357.0,
         1720146321549.0/2090206949498.0,
         3134564353537.0/4481467310338.0,
         2277821191437.0/14882151754819.0];

    def step(self, solver, dt):
        rk4a = Lserk4TimeStepper.RK4A
        rk4b = Lserk4TimeStepper.RK4B
        res_q = 0.*solver.q

        for intrk in Lserk4TimeStepper.LSERK4_STAGES:
            rhs_q = solver.compute_rhs()
            res_q = rk4a[intrk]*res_q + dt*rhs_q
            solver.q += rk4b[intrk]*res_q


