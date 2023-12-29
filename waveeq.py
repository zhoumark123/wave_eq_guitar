import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.animation import PillowWriter
from matplotlib import rc
import numba
from numba import jit
from scipy.io import wavfile
from IPython.display import Audio
#import pyaudio
import simpleaudio as sa
from pydub import AudioSegment
import copy
from multiprocessing import Process
import threading



# Nx = 101
# Nt = 500000
# L =0.7
# dx = L/(Nx-1)
# f = 220
# c = 2*L*f
# dt = 5e-6
# l=2e-5
# gamma= 1e-3# 5e-5 

Nx = 101
Nt = 500000 #500000
L =0.7
dx = L/(Nx-1)
f = 220
c = 2*L*f
dt = 5e-6 #5e-6
l=2e-5
gamma=1e-3 #2.6e-5

animation_running = False
# Initial state of the string:

# Create the initial plot with a fixed left and right endpoint
fig, ax = plt.subplots(figsize=(20,3))
ax.set_ylim(-0.01, 0.01)
line, = ax.plot(np.linspace(0,0,Nx))




#Create 2D array of $y(x, t)$


# Go through the iterative procedure:

@numba.jit("f8[:,:](f8[:,:], i8, i8, f8, f8, f8, f8)", nopython=True, nogil=True)
def compute_d(d, times, length, dt, dx, l, gamma):
    for t in range(1, times-1):
        for i in range(2, length-2):
            outer_fact = (1/(c**2 * dt**2) + gamma/(2*dt))**(-1)
            p1 = 1/dx**2 * (d[t][i-1] - 2*d[t][i] + d[t][i+1])
            p2 = 1/(c**2 * dt**2) * (d[t-1][i] - 2*d[t][i])
            p3 = gamma/(2*dt) * d[t-1][i]
            p4 = l**2 / dx**4 * (d[t][i+2] - 4*d[t][i+1] + 6*d[t][i] - 4*d[t][i-1] + d[t][i-2])
            d[t+1][i] = outer_fact * (p1 - p2 + p3 - p4)
    return d
def get_solution(y0):
    sol = np.zeros((Nt, Nx))
    sol[0] = y0
    sol[1] = y0
    sol2 = copy.deepcopy(sol)
    #plt.figure(figsize=(20,3))
    #plt.plot(y0)
    sol = compute_d(sol, Nt, Nx, dt, dx, l, gamma)
    sol_for_sound = compute_d(sol2, Nt, Nx, dt, dx, 1e-6, 2.6e-5)
    return sol, sol_for_sound
def play_sound():
    wave_obj = sa.WaveObject.from_wave_file('converted_sound.wav')
    play_obj = wave_obj.play()
    play_obj.wait_done()



def animate_string(sol):
    sound_process = threading.Thread(target=play_sound)
    def animate(i):
        if i==1 :
            sound_process.start()
        line.set_ydata(sol[i*100])
    ax.set_ylim(-0.01, 0.01)
    global ani, animation_running #global so we can stop it when we need to start a new animation
    ani = animation.FuncAnimation(fig, animate, frames=500, interval=1) #interval = 50
    animation_running = True
    #ani = animation.FuncAnimation(fig, animate, frames=250, interval=10)
    #ani.save('string.gif',writer='pillow',fps=20)
    fig.canvas.draw_idle()
    

def create_audio(sol):
    def get_integral_fast(n):
        sin_arr = np.sin(n*np.pi*np.linspace(0,1,Nx))
        return np.multiply(sol, sin_arr).sum(axis=1)

    hms = [get_integral_fast(n) for n in range(10)]

    #Add them together

    all_harmonics=True
    if all_harmonics:
        tot = sol.sum(axis=1)[::10] # all harmonics
    else:
        tot = sum(hms)[::10] # only first 10 harmonics
    tot = tot.astype(np.float32)
    #Make a WAV file
    wavfile.write('sound.wav',20000,tot)
    sound = AudioSegment.from_file('sound.wav', format='wav')
    sound = sound.set_frame_rate(44100).set_channels(1).set_sample_width(2)
    sound.export('converted_sound.wav', format='wav')
    #return tot

    
# Function to update the plot on user click
def onclick(event):
    #if animation already running, clear the plot 
    global animation_running
    if(animation_running):
        global ani
        ani._stop()
        #ax.clear()
        fig.canvas.draw()
    # Get the x and y coordinates of the click
    x_click = event.xdata
    y_click = event.ydata
     # Redraw the line with the updated coordinates
    ya = np.linspace(0, y_click, int(x_click))
    yb = np.linspace(y_click, 0, int(Nx+1-x_click))
    y0 = np.concatenate([ya, yb])

    line.set_ydata(y0)
    fig.canvas.draw()
    sol,sol_for_sound = get_solution(y0)
    tot = create_audio(sol_for_sound)
    # create a new process for the animation
    animation_process = threading.Thread(target=animate_string, args=(sol,))
    # create a new process for the sound

    # start both processes
    #playsound("sound.wav")
    animate_string(sol)
    #animation_process.start()
    #sound_process.start()
    
    
    
def on_move(event):
    global animation_running
    if(animation_running):
        return
    if event.inaxes:
        x_click = event.xdata
        y_click = event.ydata
        # Redraw the line with the updated coordinates
        ya = np.linspace(0, y_click, int(x_click))
        yb = np.linspace(y_click, 0, int(Nx+1-x_click))
        y0 = np.concatenate([ya, yb])

        line.set_ydata(y0)
        fig.canvas.draw()


def run():
    # Connect the onclick function to the plot
    binding_id = plt.connect('motion_notify_event', on_move)
    fig.canvas.mpl_connect('button_press_event', onclick)
    plt.show()
run()
# Show the plot






