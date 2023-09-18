import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

def smd_model(x,t,m,k,d,Fin):
    X = x[0]
    Xdot = x[1]
    Fs = k*X # Spring response
    Fd = d*Xdot # Damper response
    Xdotdot = (Fin - Fs - Fd)/m

    return np.array([Xdot,Xdotdot])

def step_input(time_span,signal,t_start,u_start,t_end,u_end):
    t_step = time_span[1] - time_span[0]

    index_start = np.where(time_span==t_start)[0][0]
    index_end = np.where(time_span==t_end)[0][0]

    u_start_upd = u_start + signal[index_start]
    u_end_upd = u_end + signal[index_end]

    transition_period = index_end - index_start
    slope = (u_end_upd - u_start_upd)/(t_end - t_start)

    for i in range(0,transition_period+1):
        signal[index_start+i] = slope*i*t_step

    for i in range(index_end,len(time_span)):
        signal[i] = signal[index_end]

    return signal

if __name__ == "__main__":
    print("-----------------------------------------------------------")
    print("-------------------Spring Damper Model---------------------")
    print("-----------------------------------------------------------")

    # Constants of the system
    mass = 100 # mass of the body
    k = 50 # Spring constant ( stiffness )
    d = 50 # Damping coefficient ( how fast signal dies out )

    # Simulation parameters
    sim_start = 0
    sim_end =  100
    sim_step = 0.01
    num_steps = int((sim_end - sim_start)/sim_step + 1)
    time_span = np.linspace(sim_start,sim_end,num_steps)
    control_on = True

    t_start = 4.
    t_end = 5.

    u_start = 0
    u_end = 50

    u_initial = np.full((num_steps,1),u_start)
    force_in = step_input(time_span,u_initial,t_start,u_start,t_end,u_end)

    # PID Controller Coefficients

    desired_displacement = 100
    desired_record = np.full((num_steps,1),u_start)
    desired_position = step_input(time_span,desired_record,t_start,u_start,t_end,desired_displacement)
    Kp = 50
    Kd = 100
    Ki = 0.5
    error_I = 0

    state_var = np.empty((num_steps,2))
    acceleration = np.empty((num_steps,1))
    error = np.empty((num_steps,1))

    for i in range(num_steps-1):
        inputs = (mass,k,d,force_in[i][0])
        y = odeint(smd_model,state_var[i],[0,sim_step],args = inputs)
        state_var[i+1] = y[-1]
        acceleration[i+1] = state_var[i][1]*time_span[i]
        if control_on:
            error[i] = desired_position[i] - state_var[i][0]
            error_I += error[i]
            try:
                error_D = (error[i]-error[i-1])/sim_step
            except:
                error_D = (error[i])/sim_step
            force_in[i+1] = Kp*error[i] + Ki*error_I + Kd*error_D

    print(error)
    print("Mass \t\t\t:\t",mass)    
    print("Spring Stiffness\t:\t",k)
    print("Damping coefficient \t:\t",d)
    print("Final Force input \t:\t",force_in[-1][0])
    print("Final Acceleration \t:\t",acceleration[-1][0])
    print("Final Velocity \t\t:\t",state_var[:,1][-1])
    print("Final displacement \t:\t",state_var[:,0][-1])
    print("-------------------------------------------------------------------------------")
    
    if control_on:
        plt.figure(1,figsize=(8, 8))
        plt.plot(time_span, desired_position, 'g--',linewidth=2, label = 'Desired_Position')
        plt.plot(time_span, state_var[:,0], 'r-', linewidth=1, label = 'Actual_Position')
        plt.ylabel('Displacement (Meters)')
        plt.xlabel('Time(sec)')
        plt.grid(True)
        plt.savefig('02_b_SMD_Control_PID_desired_vs_actual.png', dpi = 500)
        plt.legend(loc='best')
        plt.title("Desired displacment vs Actual displacment (controlled)")


    plt.figure(2,figsize=(8, 8))
    
    plt.subplot(4,1,1)
    plt.plot(time_span,force_in,'r-',linewidth=1,label='Step Input - Force Input')
    plt.legend(loc='best')
    plt.ylabel('Force (N)')
    plt.grid(True)
    
    plt.subplot(4,1,2)
    plt.plot(time_span,state_var[:,0],'b-',linewidth=1,label='Displacement')
    plt.legend(loc='best')
    plt.ylabel('displacement (m)')
    plt.grid(True)

    plt.subplot(4,1,3)
    plt.plot(time_span,state_var[:,1],'k:',linewidth=1,label='Velocity')
    plt.legend(loc='best')
    plt.ylabel('Velocity (m/s)')
    plt.grid(True)
    
    plt.subplot(4,1,4)
    plt.plot(time_span,acceleration,'r--',linewidth=1,label='Acceleration')
    plt.legend(loc='best')
    plt.ylabel('Acceleration (m/s²)')
    plt.grid(True)
    
    plt.xlabel('Time')
    plt.grid(True)
    plt.savefig('02_a_SMD_Modelling_Simulation.png', dpi = 500)
    plt.show()