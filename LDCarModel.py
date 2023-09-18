import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# Function to generate the required step_signal

def step_signal(t,y_in,t_start,y_start,t_end,y_end):
    ''' 
    The step signal generated here is a practical step signal. The step 
    signal's initial value is declared as y_start. Basically means that 
    at a particular time instant t_start the value of the signal starts 
    rising from y_start to a final value of y_end at a time instant t_end.
    From this time instant the value remains at y_end till the end of the
    time span. 
    '''

    y = y_in # The function arguement containing the empty array of y which will be updated
    step_size = t[1] - t[0] # Step size of the time span is calculated

    index_start = np.where(t == t_start)[0][0]
    index_end = np.where(t==t_end)[0][0]

    y_start_upd = y_start + y_in[index_start] 
    y_end_upd = y_end + y_in[index_end]
    transition_span = index_end - index_start
    slope = (y_end_upd - y_start_upd)/(t_end - t_start)

    for i in range(0,transition_span+1): # modify till index_end
        y[index_start+i] = slope*step_size*i

    for i in range(index_end,len(t)): # modify further values to constant y_end
        y[i] = y[index_end] # index_end is already updated to required value in previous loop

    return y

# The system model 

def vehicle_model(x,t,theta,c,m):
    '''
    Inputs
    --------------------------------------
    x - Current state value
    t - time span
    theta - Throttle position span
    c - caliberation factor (N/%)
    m - mass of vehicle (Kg)

    Returns
    --------------------------------------
    [xdot,xdotdot] = [velocity,acceleration]
    '''
    X = x[0]
    Xdot = x[1] # Velocity
    Xdotdot = c*theta/m

    # since the odeint function updates the values in the numpy array "state_var" the return value must be of the appropriate type np.array
    return np.array([Xdot,Xdotdot])

if __name__ == "__main__":
    
    print("-------------------------------------------------------------------------------")  
    print("---------------Simple Longitudinal Dynamics Model for a step input-------------")
    print("-------------------------------------------------------------------------------")  
    
    # Declaring constants
    C = 100
    m = 2000

    # Simulation parameters
    sim_start = 0
    sim_end = 25
    sim_step = 0.01
    num_steps = int((sim_end-sim_start)/sim_step + 1)
    time_span = np.linspace(sim_start,sim_end,num_steps)

    # Inputs

    # Time information
    t_start = 4
    t_end = 5

    # Throttle information
    y_start = 0
    y_end = 100 

    # Initial theta
    initial_theta = 0
    u_initial = np.full((num_steps,1),initial_theta)
    theta = step_signal(time_span,u_initial,t_start,y_start,t_end,y_end)

    # Results

    force = np.empty((num_steps,1))
    state_var = np.empty((num_steps,2)) # Position and velocity of car
    acceleration = np.empty((num_steps,1))

    # Simulation process
    for i in range(0,num_steps-1): # Initial state is not altered
        # Inputs for odeint args 
        inputs = (theta[i][0],C,m) # theta[i][0] because theta[i] returns a sequence with one element instead of an integer value 
        # Call odeint function
        y = odeint(vehicle_model,state_var[i],[0,sim_step],args = inputs)
        state_var[i+1] = y[-1]
        acceleration[i+1] = C*theta[i]/m
    
    print(f"Mass \t\t\t: \t{m} kg")
    print(f"Caliberation factor \t: \t{C} N/%")
    print(f"Final Throttle position : \t{theta[-1][0]} %")
    print(f"Final Acceleration \t: \t{acceleration[-1][0]} m/s^2")
    print(f"Final velocity \t\t: \t{state_var[-1][1]} m/s")
    print(f"Final displacement \t: \t{state_var[-1][0]} m")

    # PLOTS

    plt.figure(figsize=(8,8))

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
    plt.plot(time_span,state_var[:,1],'y:',linewidth=1,label="Velocity")
    plt.legend(loc="best")
    plt.ylabel("Velocity ( m/s )")
    plt.grid(True)

    plt.subplot(4,1,4)
    plt.plot(time_span,acceleration,'y:',linewidth=1,label="Velocity")
    plt.legend(loc="best")
    plt.ylabel("Acceleration ( m/s^2 )")
    plt.grid(True)

    plt.xlabel("Time")
    plt.grid(True)
    plt.savefig("Step_input_simulation.png")
    plt.show()