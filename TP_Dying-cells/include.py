import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

time_column = 7
x_column = 4
y_column = 5
track_id_column = 2
header = 4 #begin at 5th row, index 4

#to remove useless frames
def remove_consecutive_duplicates(x):
    if len(x) == 0:
        return np.array([])

    result = [x[0]]
    for i in range(1, len(x)):
        if np.abs(x[i] - x[i-1]) > 0.1:  # Use a small threshold to check for duplicates
            result.append(x[i])

    return np.array(result)


#it is not optimised but i dont care 
def remove_useless_frames(return_list):
    
    #Fin max Time
    max_time = 0
    for i in range(len(return_list)):
        m = return_list[i][0,-1]
        if m > max_time:
            max_time = m
            
    max_time = int(max_time)
        
    accepted_frames = []
    
    
    for i in range(max_time + 1) :
        #which tracks are concerned ?
        concerned_tracks_indices = []
        for j in range(len(return_list)):
            if i in return_list[j][0,:]:
                concerned_tracks_indices.append(j)
        
        #nb of concerned tracks
        n = len(concerned_tracks_indices)
        
        # print("frame", i, "concerned tracks", n)
        
        
        
        boos = []
        
        for j in concerned_tracks_indices:
            
            #locate the index of the frame
            #its not optimised, i know
            index = np.where(return_list[j][0,:] == i)[0][0]
    
            #get the track concerned
            track = return_list[j]
            x = track[1,index]
            y = track[2,index]
            
            #no comparison -> True
            if index == 0 : 
                boo = True
            else : 
                x_minus = track[1,index-1]
                y_minus = track[2,index-1]
                
                #compare with the previous frame
                boo = np.abs(x-x_minus) + np.abs(y-y_minus) > 0.3
                
            boos.append(boo)
            
            
        boos = np.array(boos)
    
        #if at least 2 tracks are above the threshold, then the frame is accepted
        if np.sum(boos) > 4 or np.sum(boos) == 4:
            # print("frame", i, "accepted")
            accepted_frames.append(i)
            
    
    accepted_frames = np.array(accepted_frames)
    # print("accepted frames", accepted_frames)
        
    return_list_not_useless = []
    
    #remove the uselessframes
    for i in range(len(return_list)):
        #remove the frames
        track = return_list[i]
        time = track[0,:]
        x = track[1,:]
        y = track[2,:]
        
        new_x = []
        new_y = []
        
        #not optimised, fuck off
        for j in range(len(x)):
            if j in accepted_frames:
                new_x.append(x[j])
                new_y.append(y[j])
                
        x = np.array(new_x)
        y = np.array(new_y)
        
        # #accepted frames len
        # print("len accepted frames", len(accepted_frames))
        # print("accepted frames", accepted_frames)   
        
        
        #n after removing useless frames
        n = len(x)
    
        #frame != time, so 
        time = np.arange(time[0],time[0] + n)
        
        return_list_not_useless.append(np.empty((3,n)))
        return_list_not_useless[i][0,:] = time
        return_list_not_useless[i][1,:] = x
        return_list_not_useless[i][2,:] = y
        
    return return_list_not_useless
        
        
def interpolate_if_skipped(x,y,t) : 
    
    t_return = []
    x_return= []
    y_return= []
    
    for i in range(len(t)-1) : 
        t_i_plus = int(t[i+1])
        t_i = int(t[i])
        
        x_i_plus = x[i+1]
        x_i = x[i]
        
        y_i_plus = y[i+1]
        y_i = y[i]
        
        #append the current point
        t_return.append(t_i)
        x_return.append(x_i)
        y_return.append(y_i)
        
        #there is a gap, e.g. t = [0,1,2,4,5,..]
        if t_i_plus-t_i > 1 : 
            #interpolate
            for j in range(t_i+1, t_i_plus) :
                t_return.append(j)
                x_return.append(x_i + (x_i_plus - x_i) * (j - t_i) / (t_i_plus - t_i))
                y_return.append(y_i + (y_i_plus - y_i) * (j - t_i) / (t_i_plus - t_i))
        
            
            
    #append the last point
    t_return.append(t[-1])
    x_return.append(x[-1])
    y_return.append(y[-1])
    #convert to numpy arrays
    x_return = np.array(x_return)
    y_return = np.array(y_return)
    t_return = np.array(t_return)

    return x_return, y_return, t_return

def import_dataset(directory,dt_directory,bugged = False): 
    """
    Input : directory
    Output : list(3x1 arrays) with arrays[:,0]=time, arrays[:,1]=x, arrays[:,2]=y
    """
    
    last_track_id = 0
    
    # Import the dataset
    dataset = pd.read_csv(directory, sep=',', header=header)
    # Convert the dataset to a numpy array
    dataset = dataset.to_numpy()


    
 
    #compute the nb of frames per track
    nb_of_frames_per_track = []
    i = 0
    
    while True:
        #NEW TRACK
        if i >= len(dataset):
            break
        
        track_id = dataset[i, track_id_column]
        last_track_id = track_id
        
        #end of the dataset
        if track_id == "":
            break
        
        #initialize
        nb_of_frames_per_track.append(1)
        
        #count the number of frames for this track
        i += 1
        while track_id == last_track_id:
            nb_of_frames_per_track[-1] += 1
            i += 1
            if i >= len(dataset):
                break
            track_id = dataset[i, track_id_column]


    #initialize the list of arrays
    return_list = []
    
    #initialize the last index
    last_index = 0
    
    
    time_at_frame = pd.read_csv(dt_directory,header=None)
    time_at_frame = time_at_frame.to_numpy()
    time_at_frame = time_at_frame[:,0]
    #the "time" column is actually the number of the frame, but it is not the real time
    #the real time at frame is in the time_at_fram array  
    
    
    for N in nb_of_frames_per_track:

        time = np.empty(N)
        x = np.empty(N)
        y = np.empty(N)
        
        for i in range(N):
            x[i] = dataset[last_index + i, x_column]
            y[i] = dataset[last_index + i, y_column]
            time[i] = np.int32(dataset[last_index + i, time_column])
            
        sorted_indices = np.argsort(time)  
        time = time[sorted_indices]
        x = x[sorted_indices]
        y = y[sorted_indices]
        
        # print("BEFORE")
        # print(time)
        
        x,y,time = interpolate_if_skipped(x,y,time)
        
        # print("AFTER")
        # print(time)
        
        # print("LEN TIME :")
        # print(len(time))
        # print("LEN TIME AT FRAME :")
        # print(len(time_at_frame))
        
        #n after removing duplicates
        n = len(x)
        return_list.append(np.empty((3,n)))
        
        return_list[-1][0,:] = time_at_frame[time.astype(int)]
        return_list[-1][1,:] = x
        return_list[-1][2,:] = y
        
        # print("AFTER 2")
        # print(time)
    
        last_index += N
    

    
      
    # for i in range(len(nb_of_frames_per_track)):
    #     #so we take the time at each frame by using the time_at_frame array
    #     # print(return_list[i][0,:].astype(int))
    #     return_list[i][0,:] = time_at_frame[return_list[i][0,:].astype(int)]
        
        
    if bugged :
        #remove useless frames
        return_list = remove_useless_frames(return_list)

    return return_list


def auto_correlation_function(vx,vy, max_lag,normalize=False):  
    """
    Computes the auto-correlation function of a 1D array v.
    v : (N,2) array, where N is the number of frames
    max_lag : int, maximum lag to compute the auto-correlation function
    Returns : ACF(t)
    """
    
    N = len(vx)
    acf = np.zeros(max_lag + 1)
    
    if normalize:
        norm = np.sqrt(vx**2 + vy**2)
        vx /= norm
        vy /= norm

    for lag in range(max_lag + 1):
        # Compute the auto-correlation for this lag
       dots = vx[:N-lag] * vx[lag:N] + vy[:N-lag] * vy[lag:N]
       acf[lag] = np.mean(dots)
    
    return acf

def MSD(x, y, max_lag):
    """
    Computes the Mean Squared Displacement (MSD) for the given x and y coordinates.
    x : (N,) array of x coordinates
    y : (N,) array of y coordinates
    max_lag : int, maximum lag to compute the MSD
    Returns : MSD(t)
    """
    
    N = len(x)
    msd = np.zeros(max_lag + 1)
    
    for lag in range(max_lag + 1):
        # Compute the MSD for this lag
        msd[lag] = np.mean((x[:N-lag] - x[lag:N])**2 + (y[:N-lag] - y[lag:N])**2)
    
    return msd


def mean_std_velocity(data_lists,Pulse_Frames,time_after_pulse=5,convolve_time=6): 
    
    avg_speeds = [[] for _ in range(len(data_lists))]
    std_speeds = [[] for _ in range(len(data_lists))]
    
    for i in range(len(data_lists)):
        data_list = data_lists[i]
        
        frame = Pulse_Frames[i]
        t = data_list[0][0]
        t_pulse = t[frame]
        
        print(f"Dataset {i+1}: Pulse at frame {frame}, time {t_pulse:.2f} seconds")
        
        for j in range(len(data_list)):
            # #crap
            # if i==1 and (j==3 or j==6) : 
            #     continue
            # Calculate the speed of the track
            x = data_list[j][1]
            y = data_list[j][2]
            t = data_list[j][0]

            dx = np.diff(x)
            dy = np.diff(y)
            dt = np.diff(t)

            speed = np.sqrt(dx**2 + dy**2) / dt
            
            speed = np.convolve(speed, np.ones(convolve_time)/convolve_time, mode='same')

            # Only keep speeds between t_pulse and t_pulse + time_after_pulse
            mask = (t[:-1] >= t_pulse) & (t[:-1] <= t_pulse + time_after_pulse)
            speed = speed[mask]
            
            if speed.size == 0:
                print(f"Warning: No speed data in selected window for track {j+1} in dataset {i+1}. Skipping this track.")
                continue
            
            # Calculate the average speed through time
            avg_speed = np.mean(speed)
            std_speed = np.std(speed)
            
            avg_speeds[i].append(avg_speed)
            std_speeds[i].append(std_speed)
    
    return avg_speeds, std_speeds

    



