import numpy as np
import Process
import time
import matplotlib.pyplot as plt
import seaborn as sns


lattice_size = 10
N = (10 ** 2) * (10 ** 4) * 2 * 3
min_temp = 0.01
max_temp = 1.5
num_samples = 1
# tot_num_samples = 41

# temperatures = np.linspace(min_temp, max_temp, num_samples)
temperatures = np.array([1])
num_particles = np.array([2000])
population_distributions = np.zeros((num_samples, lattice_size, lattice_size))
magnetization_distributions = np.zeros((num_samples, lattice_size, lattice_size))
initial_population = np.zeros((num_samples, lattice_size, lattice_size))
initial_magnetization = np.zeros((num_samples, lattice_size, lattice_size))
avg_abs_mag = []
mag_history = []
process = Process.Process(size=lattice_size, num_steps_to_steady_state=N, num_steps_in_steady_state=N, diffusion=1, epsilon=0.9)

for run in range(2):
    for j,t,p in zip(range(len(temperatures)), temperatures, num_particles):
        start_time = time.time()
        process.set_temperature(temperature=t)
        #process.set_seed(run)
        process.populate_lattice_randomly(particle_count=p)
        initial_population[j], initial_magnetization[j] = process.display_first_moments(is_initial=True)
        process.run_until_steady_state()
        #population_distributions[j], magnetization_distributions[j] = process.calculate_steady_state(visualize=True)
        #avg_abs_mag.append(np.sum(np.abs(magnetization_distributions[j])) / (lattice_size ** 2))
        # for r in range(lattice_size):
        #    correlations[j][r] = process.get_second_moment(r)
        print('{}. Temperature {:.2f} is done. Time elapsed: {:.2f}'.format(j + 1, t, time.time() - start_time))
        mag_history.append(process.get_mag_history())
plt.figure(figsize=(14, 6))
plt.plot(range(1,len(mag_history[0])+1), mag_history[0], label='run 1')
plt.plot(range(1,len(mag_history[1])+1), mag_history[1], label='run 2')
plt.xlabel('Iteration')
plt.ylabel('Magnetization per site')
plt.title('Magnetization per site vs iteration')
plt.legend()
plt.show()
np.savez('C:\\Users\\User\\Desktop\\ActiveIsingModel\\27_07_2024.npz',
         population_distributions=population_distributions, magnetization_distributions=magnetization_distributions,
         initial_population=initial_population, initial_magnetization=initial_magnetization, temperatures=temperatures,
         num_particles=num_particles, avg_abs_mag=avg_abs_mag, mag_history=mag_history)






# loaded_data = np.load('C:\\Users\\User\\Desktop\\ActiveIsingModel\\31_3_2024_test.npz')
# avg_abs_mag = np.array(loaded_data['avg_abs_mag'])
# zeros = [0, 0, 0, 0]
# x_zeros = [0, 0.5, 0.75, 1.5]
# twenties = [20, 20, 20, 20]
# plt.figure(figsize=(14, 6))
# plt.plot(x_zeros, zeros, color='black', linestyle='dashed')
# plt.plot(x_zeros, twenties, color='black', linestyle='dashed')
# plt.plot(temperatures, avg_abs_mag)
# plt.xlabel('Temperature')
# plt.ylabel('Absolute value of mean site Magnetization')
# plt.title('Site magnetization as a function of temperature')
# plt.show()


# temperatures = np.array(loaded_data['temperatures'])
# num_particles = np.array(loaded_data['num_particles'])
# densities = num_particles / (lattice_size ** 2)
# population_distributions = np.array(loaded_data['population_distributions'])
# magnetization_distributions = np.array(loaded_data['magnetization_distributions'])
# initial_population = np.array(loaded_data['initial_population'])
# initial_magnetization = np.array(loaded_data['initial_magnetization'])
#
# for j in range(len(temperatures)):
#     fig, axes = plt.subplots(1, 2, figsize=(14, 6))
#     fig.suptitle('Initial Conditions Temperature = {:.2f}, Density = {:.2f}'.format(temperatures[j], densities[j]), fontsize=16)
#
#     axes[0].set_title('Population distribution')
#     sns.heatmap(data=initial_population[j], cmap='gray',
#                           vmin=0.5 * densities[j], vmax=2 * densities[j], annot=True,
#                           cbar_kws={'label': 'Number of particles'}, ax=axes[0], fmt=".0f")
#
#     axes[1].set_title('Magnetization distribution')
#     sns.heatmap(data=initial_magnetization[j],
#                           cmap='coolwarm', annot=True, cbar_kws={'label': 'Magnetization'},
#                           ax=axes[1], fmt=".0f", vmin=-1 * densities[j], vmax=1 * densities[j])
#     plt.show()
#
#
#     fig, axes = plt.subplots(1, 2, figsize=(14, 6))
#     fig.suptitle('Temperature = {:.2f}, Density = {:.2f}'.format(temperatures[j], densities[j]), fontsize=16)
#
#     axes[0].set_title('Population distribution')
#     sns.heatmap(data=population_distributions[j], cmap='gray',
#                                 vmin=0.5 * densities[j], vmax=2 * densities[j], annot=True,
#                                 cbar_kws={'label': 'Number of particles'}, ax=axes[0], fmt=".0f")
#
#     axes[1].set_title('Magnetization distribution')
#     sns.heatmap(data=magnetization_distributions[j],
#                                 cmap='coolwarm', annot=True, cbar_kws={'label': 'Magnetization'},
#                                 ax=axes[1], fmt=".0f", vmin=-0.25 * densities[j], vmax=0.25 * densities[j])
#
#     plt.show()




# correlations = np.zeros((num_samples, lattice_size))
#correlations = np.array(loaded_data['correlations'])
'''
fig_0, axes_0 = plt.subplots(figsize=(24, 16))
for j, t in zip(range(len(temperatures)), temperatures):
    sns.lineplot(x=range(lattice_size - 1), y=correlations[j][:lattice_size - 1], label='T={:.2f}'.format(t), linewidth=5)
plt.title('Correlation length at different temperatures', fontsize=36)
plt.xticks(range(lattice_size))
plt.tick_params(axis='both', labelsize=20)
plt.xlabel('Length', fontsize=24)
plt.ylabel('Correlation', fontsize=24)
plt.legend(fontsize=20, bbox_to_anchor=(1, 1))
plt.show()
'''
