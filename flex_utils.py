from tqdm import tqdm
import numpy as np
import collections
import random
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D  # Custom legend style

# Set global font to Times New Roman
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.unicode_minus'] = False  # Fix minus sign display


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity) 

    def add(self, state, action, reward, next_state, done): 
        self.buffer.append((state, action, reward, next_state, done)) 

    def sample(self, batch_size): 
        transitions = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*transitions)
        return np.array(state), np.array(action), reward, np.array(next_state), done 

    def size(self): 
        return len(self.buffer)

    def save(self, save_path, filename='replay_buffer.pkl'):
        """Save ReplayBuffer to file in specified directory"""
        os.makedirs(save_path, exist_ok=True)  # Ensure directory exists
        filepath = os.path.join(save_path, filename)  # Build full file path
        with open(filepath, 'wb') as f:  
            pickle.dump(list(self.buffer), f)
        print(f"ReplayBuffer saved to {filepath}")

    def load(self, load_path, filename='replay_buffer.pkl'):
        """Load ReplayBuffer from file in specified directory"""
        filepath = os.path.join(load_path, filename)  # Build full file path
        with open(filepath, 'rb') as f:  # Open file in binary read mode
            data = pickle.load(f)        # Load data
            self.buffer.clear()          # Clear current buffer
            self.buffer.extend(data)     # Fill buffer
        print(f"ReplayBuffer loaded from {filepath}")

# # Old training function
# def train_off_policy_agent_td3(env, agent, num_episodes, replay_buffer, minimal_size, batch_size, writer, exploration_noise):
#     return_list = []
#     for i in range(10):
#         with tqdm(total=int(num_episodes/10), desc='Iteration %d' % i) as pbar:
#             for i_episode in range(int(num_episodes/10)):
#                 episode_return = 0
#                 state = env.reset('train')
#                 done = False
#                 while not done:
#                     action = agent.take_action(state)
#                     # add exploration noise
#                     action = np.clip(action + exploration_noise * np.random.randn(agent.action_dim), 
#                                      -agent.action_bound, agent.action_bound)

#                     next_state, reward, done, _ = env.step(action)
#                     replay_buffer.add(state, action, reward,  next_state, done)
#                     state = next_state
#                     episode_return += reward
#                     if replay_buffer.size() > minimal_size:
#                         b_s, b_a, b_r, b_ns, b_d = replay_buffer.sample(batch_size)
#                         transition_dict = {'states': b_s, 'actions': b_a, 'next_states': b_ns, 'rewards': b_r, 'dones': b_d}
#                         agent.update(transition_dict, env.i)
#                         agent.scheduler.step()  # Learning rate scheduler
#                 return_list.append(episode_return)
#                 lognum = int(num_episodes/10 * i + i_episode+1)
#                 writer.add_scalar(f"episode_reward", episode_return, lognum)
#                 #if (i_episode+1) % 10 == 0:
#                 pbar.set_postfix({'episode': '%d' % lognum, 'return': '%.3f' % np.mean(return_list[-10:])})
#                 pbar.update(1)
#     return return_list

# Training function with elite retention mechanism
def train_off_policy_agent_td3(env, agent, num_episodes, replay_buffer, minimal_size, batch_size, writer, exploration_noise, save_path=None):
    return_list = []
    return_list_elite = []
    max_avg_return = -float('inf')  # Initialize max average return
    save_path_elite = None
    
    for i in range(10):
        with tqdm(total=int(num_episodes/10), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episodes/10)):
                episode_return = 0
                state = env.reset('train')
                done = False
                while not done:
                    action = agent.take_action(state)
                    env.Tr_sol = np.column_stack((env.Tr_sol, action))
                    # add exploration noise
                    action = np.clip(action + exploration_noise * np.random.randn(agent.action_dim), 
                                     -agent.action_bound, agent.action_bound)

                    next_state, reward, done, _ = env.step(action)
                    replay_buffer.add(state, action, reward,  next_state, done)
                    state = next_state
                    episode_return += reward
                    if replay_buffer.size() > minimal_size:
                        b_s, b_a, b_r, b_ns, b_d = replay_buffer.sample(batch_size)
                        transition_dict = {'states': b_s, 'actions': b_a, 'next_states': b_ns, 'rewards': b_r, 'dones': b_d}
                        agent.update(transition_dict, env.i)
                        agent.scheduler.step()  # Learning rate scheduler
                return_list.append(episode_return)
                
                # Calculate average return of latest 10 episodes
                if len(return_list) < 10:
                    current_avg_return = np.mean(return_list)
                else:
                    current_avg_return = np.mean(return_list[-10:])
                
                # If current average return is higher, update max and save model
                if current_avg_return > max_avg_return:
                    max_avg_return = current_avg_return
                    # Save model (starting from episode 100)
                    if save_path and len(return_list) >= 100:
                        save_path_elite = os.path.join(save_path, f'elite_{int(max_avg_return)}')
                        os.makedirs(save_path_elite, exist_ok=True)
                        agent.save_model(save_path_elite)
                
                return_list_elite.append(max_avg_return)
                
                lognum = int(num_episodes/10 * i + i_episode+1)
                writer.add_scalar(f"episode_reward", episode_return, lognum)
                writer.add_scalar(f"max_average_reward", max_avg_return, lognum)
                pbar.set_postfix({'episode': '%d' % lognum, 'return': '%.3f' % np.mean(return_list[-10:])})
                pbar.update(1)
    return return_list, return_list_elite, save_path_elite


def plot_omega(t_sol, X_sol):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t_sol, y=X_sol[3], mode='lines',name='omega_x',
                             line=dict(color='orange',  # Set line color to blue
                                        width=3,      # Set line width to 2 px
                                        dash='dashdot')))
    fig.add_trace(go.Scatter(x=t_sol, y=X_sol[4], mode='lines',name='omega_y',
                             line=dict(color='rgb(74,168,52)',  # Set line color to blue
                                        width=3,      # Set line width to 2 px
                                        dash='dash')))
    fig.add_trace(go.Scatter(x=t_sol, y=X_sol[5], mode='lines',name='omega_z',
                             line=dict(color='grey',  # Set line color to blue
                                        width=3,      # Set line width to 2 px
                                        dash='dot')))

    fig.update_layout(
        width=480,  # Set chart width to 480 px
        height=360,  # Set chart height to 360 px
        title="Angular velocity over time",
        title_x=0.5,
        title_y=0.8,
        xaxis= dict(title="Timestep (s)",
                    title_standoff=1,
                    showgrid=True,  # Show X-axis grid lines
                    griddash='dot',
                    gridcolor='rgb(200, 200, 200)',  # Set X-axis grid line color to light gray
                    zeroline=False,  # Show X-axis zero line
                    zerolinecolor='rgb(50, 50, 50)',  # Set X-axis zero line color to dark gray
                    dtick=40,  # Set X-axis tick interval
                    ),
        yaxis= dict(title="Ang. Velo. (rad/s)",
                    title_standoff=1,
                    showgrid=True,  # Show X-axis grid lines
                    griddash='dot',
                    gridcolor='rgb(200, 200, 200)',  # Set Y-axis grid line color to light gray
                    zeroline=False,  # Show Y-axis zero line
                    zerolinecolor='rgb(50, 50, 50)'  # Set Y-axis zero line color to dark gray
                    ),
        legend=dict(x=0.95,  # Legend horizontal center
                    y=0.8,  # Legend vertical position slightly above center
                    xanchor='right',  # Horizontal center alignment
                    yanchor='middle',  # Bottom alignment
                    orientation='v'  # Vertical layout
                    ),
        plot_bgcolor='rgb(240, 240, 240)', # Set plot background color to light gray
        font=dict(family="Times New Roman", size=12, color="#000000"),
        # shapes=[  # Add a rectangle as title border
        # dict(
        #     type='rect',
        #     x0=0,
        #     y0=-0.09,
        #     x1=200,
        #     y1=0.085,
        #     fillcolor='rgba(0,0,0,0)',  # Border color, set to fully transparent
        #     layer='below'
        # )
        #     ],
    )
    fig.show()


def plot_p(t_sol, X_sol):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t_sol, y=X_sol[0], mode='lines',name='p1',
                             line=dict(color='orange',  # Set line color to blue
                                        width=3,      # Set line width to 2 px
                                        dash='dashdot')))
    fig.add_trace(go.Scatter(x=t_sol, y=X_sol[1], mode='lines',name='p2',
                             line=dict(color='rgb(74,168,52)',  # Set line color to blue
                                        width=3,      # Set line width to 2 px
                                        dash='dash')))
    fig.add_trace(go.Scatter(x=t_sol, y=X_sol[2], mode='lines',name='p3',
                             line=dict(color='grey',  # Set line color to blue
                                        width=3,      # Set line width to 2 px
                                        dash='dot')))
    fig.update_layout(
        width=480,  # Set chart width to 800 px
        height=360,  # Set chart height to 600 px
        title="MRP Orientation over time",
        title_x=0.5,
        title_y=0.8,
        xaxis= dict(title="Timestep (s)",
                    title_standoff=1,
                    showgrid=True,  # Show X-axis grid lines
                    griddash='dot',
                    gridcolor='rgb(200, 200, 200)',  # Set X-axis grid line color to light gray
                    zeroline=False,  # Show X-axis zero line
                    zerolinecolor='rgb(50, 50, 50)',  # Set X-axis zero line color to dark gray
                    dtick=40,  # Set X-axis tick interval
                    ),
        yaxis= dict(title="MRP",
                    title_standoff=1,
                    showgrid=True,  # Show X-axis grid lines
                    griddash='dot',
                    gridcolor='rgb(200, 200, 200)',  # Set Y-axis grid line color to light gray
                    zeroline=False,  # Show Y-axis zero line
                    zerolinecolor='rgb(50, 50, 50)'  # Set Y-axis zero line color to dark gray
                    ),
        legend=dict(x=0.95,
                    y=0.8,
                    xanchor='right',
                    yanchor='middle',
                    orientation='v'),
        plot_bgcolor='rgb(240, 240, 240)', # Set plot background color to light gray
        font=dict(family="Times New Roman", size=12, color="#000000"),  # Set font size to 12
    )
    fig.show()
def plot_Tr(t_sol, Tr_sol):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t_sol, y=Tr_sol[0], mode='lines',name='Tr1',
                             line=dict(color='orange',  # Set line color to blue
                                        width=3,      # Set line width to 2 px
                                        dash='dashdot')))
    fig.add_trace(go.Scatter(x=t_sol, y=Tr_sol[1], mode='lines',name='Tr2',
                             line=dict(color='rgb(74,168,52)',  # Set line color to blue
                                        width=3,      # Set line width to 2 px
                                        dash='dash')))
    fig.add_trace(go.Scatter(x=t_sol, y=Tr_sol[2], mode='lines',name='Tr3',
                             line=dict(color='grey',  # Set line color to blue
                                        width=3,      # Set line width to 2 px
                                        dash='dot')))
    fig.update_layout(
        width=480,  # Set chart width to 480 px
        height=360,  # Set chart height to 360 px
        title="Action torque over time",
        title_x=0.5,
        title_y=0.8,  # Align with plot_omega
        xaxis= dict(title="Timestep (s)",
                    title_standoff=1,
                    showgrid=True,  # Show X-axis grid lines
                    griddash='dot',
                    gridcolor='rgb(200, 200, 200)',  # Set X-axis grid line color to light gray
                    zeroline=False,  # Show X-axis zero line
                    zerolinecolor='rgb(50, 50, 50)',  # Set X-axis zero line color to dark gray
                    dtick=40,  # Set X-axis tick interval
                    ),
        yaxis= dict(title="Applied torque (Nm)",
                    title_standoff=1,
                    showgrid=True,  # Show X-axis grid lines
                    griddash='dot',
                    gridcolor='rgb(200, 200, 200)',  # Set Y-axis grid line color to light gray
                    zeroline=False,  # Show Y-axis zero line
                    zerolinecolor='rgb(50, 50, 50)'  # Set Y-axis zero line color to dark gray
                    ),
        legend=dict(x=0.95,  
                    y=0.8,  
                    xanchor='right',  # Horizontal center alignment
                    yanchor='middle',  # Bottom alignment
                    orientation='v'  # Vertical layout
                    ),
        plot_bgcolor='rgb(240, 240, 240)', # Set plot background color to light gray
        font=dict(family="Times New Roman", size=12, color="#000000"),  # Set font size to 12
        # shapes=[  # Add a rectangle as title border
        # dict(
        #     type='rect',
        #     x0=0,
        #     y0=-1.1,
        #     x1=200,
        #     y1=1.1,
        #     fillcolor='rgba(0,0,0,0)',  # Border color, set to fully transparent
        #     layer='below'
        # )
        #     ],
    )
    fig.show()

def plot_eta(t_sol, X_sol):
    """
    Visualize modal coordinates eta over time

    Args:
    t_sol: time series
    X_sol: state series, X_sol[6:10] are four modal coordinates (eta1, eta2, eta3, eta4)
    """
    fig = go.Figure()
    
    # Add four modal coordinates
    fig.add_trace(go.Scatter(x=t_sol, y=X_sol[6], mode='lines', name='eta1',
                             line=dict(color='orange',     # First mode
                                       width=3,
                                       dash='solid')))   # Solid line
    
    fig.add_trace(go.Scatter(x=t_sol, y=X_sol[7], mode='lines', name='eta2',
                             line=dict(color='rgb(74,168,52)',   # Second mode
                                       width=3,
                                       dash='dashdot')))
    
    fig.add_trace(go.Scatter(x=t_sol, y=X_sol[8], mode='lines', name='eta3',
                             line=dict(color='Steelblue',  # Third mode
                                       width=3,
                                       dash='dash')))
    
    fig.add_trace(go.Scatter(x=t_sol, y=X_sol[9], mode='lines', name='eta4',
                             line=dict(color='grey',     # Fourth mode
                                       width=3,
                                       dash='dot')))
    
    fig.update_layout(
        width=480,  # Set chart width to 480 px
        height=360,  # Set chart height to 360 px
        title="Modal Coordinates over time",
        title_x=0.5,
        title_y=0.8,  # Align with plot_omega
        xaxis=dict(title="Timestep (s)",
                   title_standoff=1,
                   showgrid=True,
                   griddash='dot',
                   gridcolor='rgb(200, 200, 200)',
                   zeroline=False,
                   zerolinecolor='rgb(50, 50, 50)',
                   dtick=40),
        yaxis=dict(title="Modal Coordinates",
                   title_standoff=1,
                   showgrid=True,
                   griddash='dot',
                   gridcolor='rgb(200, 200, 200)',
                   zeroline=False,
                   zerolinecolor='rgb(50, 50, 50)'),
        legend=dict(x=0.95,
                    y=0.75,
                    xanchor='right',
                    yanchor='middle',
                    orientation='v'),
        plot_bgcolor='rgb(240, 240, 240)', # Set plot background color to light gray
        font=dict(family="Times New Roman", size=12, color="#000000"),  # Set font size to 12
    )
    fig.show()



def plot_states(t_sol, X_sol, Tr_sol, MDPI_style=False):
    """
    Matplotlib: plot state trajectories

    Args:
        t_sol: time series array (1D)
        X_sol: state series array (10xN), containing [p1-p3, omega_x-omega_z, eta1-eta4]
        Tr_sol: control torque series array (3xN), containing [Tr1-Tr3]
        MDPI_style: False=2x2 layout, True=1x4 layout (enlarged legend/title, x-axis suffix (s))
    """
    if not MDPI_style:
        # Core change: 2x2 layout, adjusted canvas size for typesetting
        fig, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)

        # Transparent canvas background (unchanged)
        fig.patch.set_alpha(0.0)
        fig.patch.set_facecolor('none')

        # --- 1. Row 1, Col 1: MRP Attitude ((i) Attitude) ---
        ax_a = axes[0, 0]
        # Legend with Greek letters p₁,p₂,p₃
        ax_a.plot(t_sol, X_sol[0], color='#008FD5', linewidth=1.5, linestyle='solid', label=r'$p_1$')
        ax_a.plot(t_sol, X_sol[1], color='#582A8A', linewidth=1.5, linestyle='solid', label=r'$p_2$')
        ax_a.plot(t_sol, X_sol[2], color='#EF3F22', linewidth=1.5, linestyle='solid', label=r'$p_3$')

        ax_a.set_xlabel("Time (s)", fontsize=14)
        ax_a.set_ylabel("MRP", fontsize=14)
        ax_a.grid(True, linestyle='dotted', color='#c8c8c8', alpha=1.0)
        ax_a.set_axisbelow(True)
        for spine in ax_a.spines.values():
            spine.set_visible(True)
            spine.set_color('#323232')
            spine.set_linewidth(1.5)
        ax_a.xaxis.set_major_locator(plt.MultipleLocator(40))
        ax_a.legend(loc='upper right', frameon=True, framealpha=0.8,
                   edgecolor='lightgrey', facecolor='white')
        ax_a.set_facecolor('none')
        ax_a.annotate('(i) Attitude', xy=(0.5, -0.15), xycoords='axes fraction',
                     ha='center', va='top', fontsize=16)

        # --- 2. Row 1, Col 2: Angular Velocity ((ii) Angular Velocity) ---
        ax_b = axes[0, 1]
        # Legend with Greek letters ωₓ,ωᵧ,ω_z
        ax_b.plot(t_sol, X_sol[3], color='#008FD5', linewidth=1.5, linestyle='solid', label=r'$\omega_x$')
        ax_b.plot(t_sol, X_sol[4], color='#582A8A', linewidth=1.5, linestyle='solid', label=r'$\omega_y$')
        ax_b.plot(t_sol, X_sol[5], color='#EF3F22', linewidth=1.5, linestyle='solid', label=r'$\omega_z$')

        ax_b.set_xlabel("Time (s)", fontsize=14)
        ax_b.set_ylabel("Ang. Velo. (rad/s)", fontsize=14)
        ax_b.grid(True, linestyle='dotted', color='#c8c8c8', alpha=1.0)
        ax_b.set_axisbelow(True)
        for spine in ax_b.spines.values():
            spine.set_visible(True)
            spine.set_color('#323232')
            spine.set_linewidth(1.5)
        ax_b.xaxis.set_major_locator(plt.MultipleLocator(40))
        ax_b.legend(loc='upper right', frameon=True, framealpha=0.8,
                   edgecolor='lightgrey', facecolor='white')
        ax_b.set_facecolor('none')
        ax_b.annotate('(ii) Angular Velocity', xy=(0.5, -0.15), xycoords='axes fraction',
                     ha='center', va='top', fontsize=16)

        # --- 3. Row 2, Col 1: Modal Displacement ((iii) Modal Displacement) ---
        ax_c = axes[1, 0]
        # Legend with Greek letters η₁-η₄
        ax_c.plot(t_sol, X_sol[6], color='#008FD5', linewidth=1.5, linestyle='solid', label=r'$\eta_1$')
        ax_c.plot(t_sol, X_sol[7], color='#582A8A', linewidth=1.5, linestyle='solid', label=r'$\eta_2$')
        ax_c.plot(t_sol, X_sol[8], color='#EF3F22', linewidth=1.5, linestyle='solid', label=r'$\eta_3$')
        ax_c.plot(t_sol, X_sol[9], color='#FFA503', linewidth=1.5, linestyle='solid', label=r'$\eta_4$')

        ax_c.set_xlabel("Time (s)", fontsize=14)
        ax_c.set_ylabel("Modal Coordinates", fontsize=14)
        ax_c.grid(True, linestyle='dotted', color='#c8c8c8', alpha=1.0)
        ax_c.set_axisbelow(True)
        for spine in ax_c.spines.values():
            spine.set_visible(True)
            spine.set_color('#323232')
            spine.set_linewidth(1.5)
        ax_c.xaxis.set_major_locator(plt.MultipleLocator(40))
        ax_c.legend(loc='upper right', frameon=True, framealpha=0.8,
                   edgecolor='lightgrey', facecolor='white')
        ax_c.set_facecolor('none')
        ax_c.annotate('(iii) Modal Displacement', xy=(0.5, -0.15), xycoords='axes fraction',
                     ha='center', va='top', fontsize=16)

        # --- 4. Row 2, Col 2: Torque Output ((iv) Torque Output) ---
        ax_d = axes[1, 1]
        ax_d.plot(t_sol, Tr_sol[0], color='#008FD5', linewidth=1.5, linestyle='solid', label=r'$u_1$')
        ax_d.plot(t_sol, Tr_sol[1], color='#582A8A', linewidth=1.5, linestyle='solid', label=r'$u_2$')
        ax_d.plot(t_sol, Tr_sol[2], color='#EF3F22', linewidth=1.5, linestyle='solid', label=r'$u_3$')

        ax_d.set_xlabel("Time (s)", fontsize=14)
        ax_d.set_ylabel("Applied torque (Nm)", fontsize=14)
        ax_d.grid(True, linestyle='dotted', color='#c8c8c8', alpha=1.0)
        ax_d.set_axisbelow(True)
        for spine in ax_d.spines.values():
            spine.set_visible(True)
            spine.set_color('#323232')
            spine.set_linewidth(1.5)
        ax_d.xaxis.set_major_locator(plt.MultipleLocator(40))
        ax_d.legend(loc='upper right', frameon=True, framealpha=0.8,
                   edgecolor='lightgrey', facecolor='white')
        ax_d.set_facecolor('none')
        ax_d.annotate('(iv) Torque Output', xy=(0.5, -0.15), xycoords='axes fraction',
                     ha='center', va='top', fontsize=16)

        # Show plot
        plt.show()

    else:
        # MDPI style: 1x4 layout + x-axis suffix (s) + enlarged legend/title
        fig, axes = plt.subplots(1, 4, figsize=(28, 6), constrained_layout=True)

        fig.patch.set_alpha(0.0)
        fig.patch.set_facecolor('none')

        tick_fmt = plt.FuncFormatter(lambda x, pos: f'{int(x)} (s)')
        legend_fs = 24
        title_fs = 32
        label_fs = 24
        tick_fs = 20
        plot_lw = 2.0
        legend_kw = {'prop': {'size': legend_fs}}

        # --- 1. MRP Attitude ((i) Attitude) ---
        ax_a = axes[0]
        ax_a.plot(t_sol, X_sol[0], color='#008FD5', linewidth=plot_lw, linestyle='solid', label=r'$p_1$')
        ax_a.plot(t_sol, X_sol[1], color='#582A8A', linewidth=plot_lw, linestyle='solid', label=r'$p_2$')
        ax_a.plot(t_sol, X_sol[2], color='#EF3F22', linewidth=plot_lw, linestyle='solid', label=r'$p_3$')

        ax_a.set_xlabel("")
        ax_a.set_ylabel("MRP", fontsize=label_fs)
        ax_a.grid(True, linestyle='dotted', color='#c8c8c8', alpha=1.0)
        ax_a.set_axisbelow(True)
        for spine in ax_a.spines.values():
            spine.set_visible(True)
            spine.set_color('#323232')
            spine.set_linewidth(plot_lw)
        ax_a.tick_params(labelsize=tick_fs)
        ax_a.xaxis.set_major_locator(plt.MultipleLocator(40))
        ax_a.xaxis.set_major_formatter(tick_fmt)
        ax_a.legend(loc='upper right', frameon=True, framealpha=0.8,
                   edgecolor='lightgrey', facecolor='white', **legend_kw)
        ax_a.set_facecolor('none')
        ax_a.annotate('(i) Attitude', xy=(0.5, -0.10), xycoords='axes fraction',
                     ha='center', va='top', fontsize=title_fs)

        # --- 2. Angular Velocity ((ii) Angular Velocity) ---
        ax_b = axes[1]
        ax_b.plot(t_sol, X_sol[3], color='#008FD5', linewidth=plot_lw, linestyle='solid', label=r'$\omega_x$')
        ax_b.plot(t_sol, X_sol[4], color='#582A8A', linewidth=plot_lw, linestyle='solid', label=r'$\omega_y$')
        ax_b.plot(t_sol, X_sol[5], color='#EF3F22', linewidth=plot_lw, linestyle='solid', label=r'$\omega_z$')

        ax_b.set_xlabel("")
        ax_b.set_ylabel("Ang. Velo. (rad/s)", fontsize=label_fs)
        ax_b.grid(True, linestyle='dotted', color='#c8c8c8', alpha=1.0)
        ax_b.set_axisbelow(True)
        for spine in ax_b.spines.values():
            spine.set_visible(True)
            spine.set_color('#323232')
            spine.set_linewidth(plot_lw)
        ax_b.tick_params(labelsize=tick_fs)
        ax_b.xaxis.set_major_locator(plt.MultipleLocator(40))
        ax_b.xaxis.set_major_formatter(tick_fmt)
        ax_b.legend(loc='upper right', frameon=True, framealpha=0.8,
                   edgecolor='lightgrey', facecolor='white', **legend_kw)
        ax_b.set_facecolor('none')
        ax_b.annotate('(ii) Angular Velocity', xy=(0.5, -0.10), xycoords='axes fraction',
                     ha='center', va='top', fontsize=title_fs)

        # --- 3. Modal Displacement ((iii) Modal Displacement) ---
        ax_c = axes[2]
        ax_c.plot(t_sol, X_sol[6], color='#008FD5', linewidth=plot_lw, linestyle='solid', label=r'$\eta_1$')
        ax_c.plot(t_sol, X_sol[7], color='#582A8A', linewidth=plot_lw, linestyle='solid', label=r'$\eta_2$')
        ax_c.plot(t_sol, X_sol[8], color='#EF3F22', linewidth=plot_lw, linestyle='solid', label=r'$\eta_3$')
        ax_c.plot(t_sol, X_sol[9], color='#FFA503', linewidth=plot_lw, linestyle='solid', label=r'$\eta_4$')

        ax_c.set_xlabel("")
        ax_c.set_ylabel("Modal Coordinates", fontsize=label_fs)
        ax_c.grid(True, linestyle='dotted', color='#c8c8c8', alpha=1.0)
        ax_c.set_axisbelow(True)
        for spine in ax_c.spines.values():
            spine.set_visible(True)
            spine.set_color('#323232')
            spine.set_linewidth(plot_lw)
        ax_c.tick_params(labelsize=tick_fs)
        ax_c.xaxis.set_major_locator(plt.MultipleLocator(40))
        ax_c.xaxis.set_major_formatter(tick_fmt)
        ax_c.legend(loc='upper right', frameon=True, framealpha=0.8,
                   edgecolor='lightgrey', facecolor='white', **legend_kw)
        ax_c.set_facecolor('none')
        ax_c.annotate('(iii) Modal Displacement', xy=(0.5, -0.10), xycoords='axes fraction',
                     ha='center', va='top', fontsize=title_fs)

        # --- 4. Torque Output ((iv) Torque Output) ---
        ax_d = axes[3]
        ax_d.plot(t_sol, Tr_sol[0], color='#008FD5', linewidth=plot_lw, linestyle='solid', label=r'$u_1$')
        ax_d.plot(t_sol, Tr_sol[1], color='#582A8A', linewidth=plot_lw, linestyle='solid', label=r'$u_2$')
        ax_d.plot(t_sol, Tr_sol[2], color='#EF3F22', linewidth=plot_lw, linestyle='solid', label=r'$u_3$')

        ax_d.set_xlabel("")
        ax_d.set_ylabel("Applied torque (Nm)", fontsize=label_fs)
        ax_d.grid(True, linestyle='dotted', color='#c8c8c8', alpha=1.0)
        ax_d.set_axisbelow(True)
        for spine in ax_d.spines.values():
            spine.set_visible(True)
            spine.set_color('#323232')
            spine.set_linewidth(plot_lw)
        ax_d.tick_params(labelsize=tick_fs)
        ax_d.xaxis.set_major_locator(plt.MultipleLocator(40))
        ax_d.xaxis.set_major_formatter(tick_fmt)
        ax_d.legend(loc='upper right', frameon=True, framealpha=0.8,
                   edgecolor='lightgrey', facecolor='white', **legend_kw)
        ax_d.set_facecolor('none')
        ax_d.annotate('(iv) Torque Output', xy=(0.5, -0.10), xycoords='axes fraction',
                     ha='center', va='top', fontsize=title_fs)

        plt.show()


def plot_d(t_sol, d_sol):
    """
    Visualize disturbance over time (MDPI style)

    Args:
    t_sol: time series
    d_sol: disturbance series, three components (d1, d2, d3)
    """
    legend_fs = 22
    title_fs = 24
    label_fs = 16
    tick_fs = 18

    fig, ax = plt.subplots(1, 1, figsize=(7, 5), constrained_layout=True)

    fig.patch.set_alpha(0.0)
    fig.patch.set_facecolor('none')

    tick_fmt = plt.FuncFormatter(lambda x, pos: f'{int(x)} (s)')

    ax.plot(t_sol, d_sol[0], color='#008FD5', linewidth=1.5, linestyle='solid', label=r'$d_1$')
    ax.plot(t_sol, d_sol[1], color='#582A8A', linewidth=1.5, linestyle='solid', label=r'$d_2$')
    ax.plot(t_sol, d_sol[2], color='#EF3F22', linewidth=1.5, linestyle='solid', label=r'$d_3$')

    ax.set_xlabel("")
    ax.set_ylabel("Disturbance", fontsize=label_fs)
    ax.grid(True, linestyle='dotted', color='#c8c8c8', alpha=1.0)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color('#323232')
        spine.set_linewidth(1.5)
    ax.tick_params(labelsize=tick_fs)
    ax.xaxis.set_major_locator(plt.MultipleLocator(40))
    ax.xaxis.set_major_formatter(tick_fmt)
    ax.legend(loc='upper right', frameon=True, framealpha=0.8,
              edgecolor='lightgrey', facecolor='white', prop={'size': legend_fs})
    ax.set_facecolor('none')
    ax.annotate('Disturbance', xy=(0.5, -0.10), xycoords='axes fraction',
                ha='center', va='top', fontsize=title_fs)

    plt.show()


def performance_old(t_sol, X_sol):
    """
    Comprehensive performance analysis: MRP, angular velocity, and modal vibration convergence

    Args:
    t_sol: time series
    X_sol: state series, X_sol[0:3]=MRP, X_sol[3:6]=ang. velo., X_sol[6:10]=modal vibration
    """
    # --- MRP Convergence ---
    p1, p2, p3 = X_sol[0], X_sol[1], X_sol[2]
    mrp_norm = np.maximum(np.maximum(np.abs(p1), np.abs(p2)), np.abs(p3))

    converge_time_01 = None
    for i in range(len(mrp_norm)):
        if all(val <= 0.01 for val in mrp_norm[i:]):
            converge_time_01 = t_sol[i]
            break
    converge_time_005 = None
    for i in range(len(mrp_norm)):
        if all(val <= 0.005 for val in mrp_norm[i:]):
            converge_time_005 = t_sol[i]
            break

    print(f"Earliest time MRP converges within 0.01 (s): {converge_time_01}")
    print(f"Earliest time MRP converges within 0.005 (s): {converge_time_005}")

    # --- Angular Velocity Convergence ---
    omega_x, omega_y, omega_z = X_sol[3], X_sol[4], X_sol[5]
    omega_norm = np.maximum(np.maximum(np.abs(omega_x), np.abs(omega_y)), np.abs(omega_z))

    converge_time_01 = None
    for i in range(len(omega_norm)):
        if all(val <= 0.01 for val in omega_norm[i:]):
            converge_time_01 = t_sol[i]
            break
    converge_time_005 = None
    for i in range(len(omega_norm)):
        if all(val <= 0.005 for val in omega_norm[i:]):
            converge_time_005 = t_sol[i]
            break

    print(f"Earliest time angular velocity converges within 0.01 (s): {converge_time_01}")
    print(f"Earliest time angular velocity converges within 0.005 (s): {converge_time_005}")

    # --- Modal Vibration Convergence ---
    eta1, eta2, eta3, eta4 = X_sol[6], X_sol[7], X_sol[8], X_sol[9]
    eta_norm = np.maximum(np.maximum(np.maximum(np.abs(eta1), np.abs(eta2)), np.abs(eta3)), np.abs(eta4))
    max_eta_value = round(np.max(eta_norm), 3)

    converge_time_01 = None
    for i in range(len(eta_norm)):
        if all(val <= 0.01 for val in eta_norm[i:]):
            converge_time_01 = t_sol[i]
            break
    converge_time_005 = None
    for i in range(len(eta_norm)):
        if all(val <= 0.005 for val in eta_norm[i:]):
            converge_time_005 = t_sol[i]
            break

    print(f"Maximum modal displacement: {max_eta_value}")
    print(f"Earliest time modal displacement converges within 0.01 (s): {converge_time_01}")
    print(f"Earliest time modal displacement converges within 0.005 (s): {converge_time_005}")
    
def performance_test6(t_sol, X_sol):
    """
    Compute MRP steady-state error

    Args:
    t_sol: time series
    X_sol: state series, X_sol[0:3] are MRP components (p1, p2, p3)
    """
    # Get three MRP components
    p1 = X_sol[0]
    p2 = X_sol[1]
    p3 = X_sol[2]

    # Compute inf-norm of final MRP vector (max absolute value)
    final_p_norm = max(abs(p1[-1]), abs(p2[-1]), abs(p3[-1]))

    print(f"Steady-state error of p (max abs value at final timestep): {final_p_norm}")

    return final_p_norm


def performance_steadystate(t_sol, X_sol, Tr_sol, t_ss=160):
    """
    Steady-state quantitative analysis (t > t_ss)

    Args:
    t_sol: time series
    X_sol: state series, X_sol[0:3]=MRP, X_sol[3:6]=ang. velo., X_sol[6:10]=modal coordinates
    Tr_sol: control torque series (3xN)
    t_ss: steady-state start time, default 160s
    """
    idx_ss = np.where(t_sol >= t_ss)[0]
    if len(idx_ss) == 0:
        print(f"Warning: simulation time < {t_ss}s, cannot compute steady-state metrics")
        return

    p_ss = X_sol[0:3, idx_ss]
    ss_mrp_norm = np.max(np.max(np.abs(p_ss), axis=0))

    omega_ss = X_sol[3:6, idx_ss]
    ss_omega_norm = np.max(np.max(np.abs(omega_ss), axis=0))

    eta_ss = X_sol[6:10, idx_ss]
    ss_eta_rms = np.sqrt(np.mean(np.sum(eta_ss**2, axis=0)))

    torque_ss = Tr_sol[:, idx_ss]
    ss_torque_rms = np.sqrt(np.mean(np.sum(torque_ss**2, axis=0)))

    print("=" * 55)
    print(f"  Steady-state Metrics (t > {t_ss}s)")
    print("=" * 55)
    print(f"  SS MRP ||p||_inf          : {ss_mrp_norm:.6f}")
    print(f"  SS Ang. Velo. ||ω||_inf   : {ss_omega_norm:.6f} rad/s")
    print(f"  SS Modal Disp. RMS        : {ss_eta_rms:.6f}")
    print(f"  SS Control Torque RMS     : {ss_torque_rms:.6f} Nm")
    print("=" * 55)

    return {
        'ss_mrp_norm': ss_mrp_norm,
        'ss_omega_norm': ss_omega_norm,
        'ss_eta_rms': ss_eta_rms,
        'ss_torque_rms': ss_torque_rms,
    }


def performance_transient(t_sol, X_sol, Tr_sol):
    """
    Transient / full-trajectory quantitative analysis

    Args:
    t_sol: time series
    X_sol: state series, X_sol[0:3]=MRP, X_sol[3:6]=ang. velo., X_sol[6:10]=modal coordinates
    Tr_sol: control torque series (3xN)
    """
    # MRP convergence times
    p_all = X_sol[0:3, :]
    mrp_norm = np.max(np.abs(p_all), axis=0)

    converge_time_01 = None
    for i in range(len(mrp_norm)):
        if all(val <= 0.01 for val in mrp_norm[i:]):
            converge_time_01 = t_sol[i]
            break

    converge_time_005 = None
    for i in range(len(mrp_norm)):
        if all(val <= 0.005 for val in mrp_norm[i:]):
            converge_time_005 = t_sol[i]
            break

    # Peak angular velocity norm (entire trajectory)
    omega_all = X_sol[3:6, :]
    omega_inf_max = np.max(np.max(np.abs(omega_all), axis=0))

    # RMS of modal displacement (entire trajectory)
    eta_all = X_sol[6:10, :]
    eta_rms = np.sqrt(np.mean(np.sum(eta_all**2, axis=0)))

    # Torque RMS (full trajectory)
    torque_rms = np.sqrt(np.mean(np.sum(Tr_sol**2, axis=0)))

    print("=" * 55)
    print("  Transient / Full-Trajectory Metrics")
    print("=" * 55)
    print(f"  MRP converge to 0.01 time  : {converge_time_01}s")
    print(f"  MRP converge to 0.005 time : {converge_time_005}s")
    print(f"  Full Ang. Velo. ||ω||_inf  : {omega_inf_max:.6f} rad/s")
    print(f"  Full Modal Disp. RMS       : {eta_rms:.6f}")
    print(f"  Full Control Torque RMS    : {torque_rms:.6f} Nm")
    print("=" * 55)

    return {
        'converge_time_01': converge_time_01,
        'converge_time_005': converge_time_005,
        'omega_inf_max': omega_inf_max,
        'eta_rms': eta_rms,
        'torque_rms': torque_rms,
    }