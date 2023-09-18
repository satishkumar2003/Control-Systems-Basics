import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

def step_signal(t,y_in,t_start,y_start,t_end,y_end):
    y = y_in
    time_step = t[1]-t[0]

    index_start = np.where(t==t_start)[0][0]
    index_end = np.where(t==t_end)[0][0]

    y_start_upd = y_start + y[index_start]
    y_end_upd = y_end + y[index_end]
    transition_span = index_end - index_start
    slope = (y_end_upd - y_start_upd)/(t_end - t_start)

    for i in range(0,transition_span+1):
        y[index_start+i] = i*time_step*slope

    for i in range(index_end,len(t)):
        y[i] = y[index_end]

    return y


def vehicle_model(x,t,theta,c,m):
    X = x[0]
    Xdot = x[1]
    Xdotdot = c*theta/m

    return np.array([Xdot, Xdotdot])

if __name__ == "__main__":
    
    print("-------------------------------------------------------------------------------")  
    print("---------------Simple Longitudinal Dynamics Model for a step input-------------")
    print("-------------------------------------------------------------------------------")  
    
    # Declaring constants
    C = 100
    m = 2000

    # Simulation parameters
    sim_start = 0
    sim_end = 250
    sim_step = 0.01
    num_steps = int((sim_end-sim_start)/sim_step + 1)
    time_span = np.linspace(sim_start,sim_end,num_steps)
    control_on = True

    #Throttle information

    t_start = 4.
    t_end = 5.  

    y_start = 0
    y_end = 100

    throttle_constraints = [-100,100] # min for breaking and max for full throttle

    initial_theta = 0
    u_initial = np.full((num_steps,1),initial_theta)
    theta = step_signal(time_span,u_initial,t_start,y_start,t_end,y_end)

    # Controller parameters
    desired_displacement = 1000
    y_desired = np.full((num_steps,1),y_start)
    desired_position = step_signal(time_span,y_desired,t_start,y_start,t_end,desired_displacement)

    # PID Controller

    Kp = 10 # Increases overshoot
    Ki = 0.00001 # Integral error makes controller slower
    Kd = 50
    error_integrated = 0

    state_var = np.empty((num_steps,2))
    acceleration = np.empty((num_steps,1))
    error = np.empty((num_steps,1))

    for i in range(0,num_steps-1):
        inputs = theta[i][0],C,m
        y = odeint(vehicle_model,state_var[i],[0,sim_step],args = inputs)
        state_var[i+1] = y[-1]
        acceleration[i+1] = C*theta[i]/m

        if control_on:
            error[i] = desired_position[i] - state_var[i][0]
            error_integrated += error[i]
            try:
                error_differential = (error[i] - error[i-1])/sim_step
            except:
                error_differential = (error[i] - 0)/sim_step
            theta_in = Kp*error[i] + Ki*error_integrated + Kd*error_differential
            if theta_in < throttle_constraints[0]:
                theta_in = -100
            elif theta_in > throttle_constraints[1]:
                theta_in = 100
            theta[i+1] = theta_in

    print(f"Mass \t\t\t\t\t: \t{m} kg")
    print(f"Caliberation factor \t\t\t: \t{C} N/%")
    print(f"Final Throttle position \t\t: \t{theta[-1][0]} %")
    print(f"Final Acceleration \t\t\t: \t{acceleration[-1][0]} m/s^2")
    print(f"Final velocity \t\t\t\t: \t{state_var[-1][1]} m/s")
    print(f"Final displacement \t\t\t: \t{state_var[-1][0]} m")
    print(f"Steady state error for displacement\t: \t{abs(state_var[-1][0]-desired_displacement)}")

    # PLOTS

    if control_on:
        plt.figure(1,figsize=(8,8))
        plt.plot(time_span,desired_position,"g--",linewidth=2,label="Desired_displacement")
        plt.plot(time_span,state_var[:,0],'r-',linewidth=1,label="Actual_displacement")
        plt.ylabel("Displacement (m)")
        plt.xlabel("Time (s)")
        plt.grid(True)
        plt.savefig("PID_Control.png")
        plt.legend(loc="best")
        plt.title("Desired displacement vs Actual displacement")

    plt.figure(2,figsize=(8,8))

    plt.subplot(4,1,1)
    plt.plot(time_span,theta,'r-',linewidth=1,label="Step Input - Theta")
    plt.legend(loc = "best")
    plt.ylabel("Theta ( % )")
    plt.grid(True)

    plt.subplot(4,1,2)
    plt.plot(time_span,state_var[:,0],'b-',linewidth=1,label="Displacement")
    plt.legend(loc="best")
    plt.ylabel("Displacement ( m )")
    plt.grid(True)

    plt.subplot(4,1,3)
    plt.plot(time_span,state_var[:,1],'g-',linewidth=1,label="Velocity")
    plt.legend(loc="best")
    plt.ylabel("Velocity ( m/s )")
    plt.grid(True)

    plt.subplot(4,1,4)
    plt.plot(time_span,acceleration,'m-',linewidth=1,label="Acceleration")
    plt.legend(loc="best")
    plt.ylabel("Acceleration ( m/s^2 )")
    plt.grid(True)

    plt.xlabel("Time")
    plt.grid(True)
    plt.savefig("Step_input_simulation_with_PID.png")
    plt.show()