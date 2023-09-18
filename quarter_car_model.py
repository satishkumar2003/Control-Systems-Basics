import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

def step_input(t,y,t_start,y_start,t_end,y_end):
    step_size = t[1]-t[0]
    index_start = np.where(t==t_start)[0][0]
    index_end = np.where(t==t_end)[0][0]
    y_start_upd = y_start + y[index_start]
    y_end_upd = y_end + y[index_end]
    transition_period = index_end-index_start
    slope = (y_end_upd - y_start_upd)/(t_end-t_start)

    for i in range(0,transition_period+1):
        y[index_start+i] = slope*i*step_size
    
    for i in range(index_end,len(t)):
        y[i] = y[index_end]

    return y

def quarter_car_model(x,t,mass_chassis,mass_wheel,ks,ds,kt,Zr):
    '''
    x - current system state values
    t - timespan
    mass_chassis - mass of chassis
    mass_wheel - mass of wheel
    ks - spring stiffness
    ds - damping coefficient
    kt - spring stiffness for wheel
    Zr - Height from road

    Zwdot - Vertical wheel speed
    Zwdotdot - Vertical wheel position
    Zcdot - Vertical chassic speed
    Zcdotdot - Vertical chassis position
    '''
    Zw = x[0]
    Zwdot = x[1]
    Zc = x[2]
    Zcdot = x[3]

    g = 9.81
    R_tire = 0.3 # Tire radius
    FL = 0.4 # Free length of spring
    fs = Zw-Zc
    fsdot = Zwdot - Zcdot
    ft = Zr-Zw+FL
    Fs = ks*fs+ds*fsdot
    Ft = kt*ft+R_tire

    Zwdotdot = (Ft-Fs-mass_wheel*g)/mass_wheel
    Zcdotdot = (Fs-mass_chassis*g)/mass_chassis

    return np.array([Zwdot,Zwdotdot,Zcdot,Zcdotdot])


if __name__ == "__main__":
    print("-------------------------------------------------------------------------------")  
    print("---------------- Quarter Car model - Modelling and Simulation -----------------")     # Simulation name
    print("-------------------------------------------------------------------------------")
    

    # Constants 
    mass_chassis = 300
    mass_wheel = 30
    ks = 20000 # Spring stiffness N/m
    ds = 1500 # Damping coefficient Ns/m
    kt = 220000 # Stiffness of tire N/m

    # Simulation setup
    sim_start = 0
    sim_end = 100
    sim_step = 0.01
    num_steps = int((sim_end-sim_start)/sim_step+1)
    time_span = np.linspace(sim_start,sim_end,num_steps)
    control_on = False

    # Inputs
    t_start = 20.
    t_end = 25.
    u_start = 0
    u_end = 1
    u_initial = np.full((num_steps,1),u_start,'float')
    Zr_in = step_input(time_span,u_initial,t_start,u_start,t_end,u_end)

    # Result collection
    state_var = np.empty((num_steps,4))
    state_var[0] = np.array([0.3,0,0.7,0]) # 0.3m is initial position of center of wheel and 0.4m is the free spring length so the chassis beginning is at 0.7m
    acc_wheel = np.empty((num_steps,1))
    acc_chassis = np.empty((num_steps,1))
    error = np.empty((num_steps,1))

    for i in range(num_steps-1):
        inputs = (mass_chassis,mass_wheel,ks,ds,kt,Zr_in[i][0])
        y = odeint(quarter_car_model,state_var[i],[0,sim_step],args = inputs)
        state_var[i+1] = y[-1]
        acc_wheel[i+1] = state_var[i][0]/time_span[i]
        acc_chassis[i+1] = state_var[i][2]/time_span[i]


    print("Mass of chassis\t\t\t\t:\t",mass_chassis)    
    print("Spring Stiffness of suspension\t\t:\t",ks)
    print("Damping coefficient of suspension\t:\t",ds)
    print("Mass of wheel\t\t\t\t:\t",mass_wheel)    
    print("Spring Stiffness of wheel\t\t:\t",kt)
   
    #print("Final Control Input \t\t\t:\t",ds_in)
    
    print("Final Chassis Acceleration \t\t:\t",acc_chassis[-1][0])
    print("Final Chassis Velocity \t\t\t:\t",state_var[:,3][-1])
    print("Final Chassis displacement \t\t:\t",state_var[:,2][-1])
        
    print("Final Wheel Acceleration \t\t:\t",acc_wheel[-1][0])
    print("Final Wheel Velocity \t\t\t:\t",state_var[:,1][-1])
    print("Final Wheel displacement \t\t:\t",state_var[:,0][-1])
    print("-------------------------------------------------------------------------------")
    ###################### Controlled plant - Plots ################
    # Plot of the controlled plant (desired vs actual )

    ###################### Summary - Plots #########################
    # Summary plot / additional plot
    plt.figure(figsize=(8, 8))
    
    plt.subplot(4,1,1)
    plt.plot(time_span,Zr_in,'k-',linewidth=1,label='Step Input - Force Input')
    plt.legend(loc='best')
    plt.ylabel('Road Height (m)')
    plt.grid(True)
    
    plt.subplot(4,1,2)
    plt.plot(time_span,state_var[:,0],'b-',linewidth=1,label='Displacement_wheel')
    plt.plot(time_span,state_var[:,2],'r-',linewidth=1,label='Displacement_chassis')
    plt.legend(loc='best')
    plt.ylabel('displacement (m)')
    plt.grid(True)

    plt.subplot(4,1,3)
    plt.plot(time_span,state_var[:,1],'b:',linewidth=1,label='Velocity_wheel')
    plt.plot(time_span,state_var[:,3],'r:',linewidth=1,label='Velocity_chassis')
    plt.legend(loc='best')
    plt.ylabel('Velocity (m/s)')
    plt.grid(True)
    
    plt.subplot(4,1,4)
    plt.plot(time_span,acc_wheel     ,'b--',linewidth=1,label='Acceleration_wheel')
    plt.plot(time_span,acc_chassis   ,'r--',linewidth=1,label='Acceleration_chassis')
    plt.legend(loc='best')
    plt.ylabel('Acceleration (m/s²)')
    plt.grid(True)
    
    plt.xlabel('Time')
    plt.grid(True)
    plt.savefig('03_a_QCM_Modelling_Simulation.png', dpi = 500)
    plt.show()