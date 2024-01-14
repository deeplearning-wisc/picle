import matplotlib.pyplot as plt
import seaborn as sns

# sim_data = [65.5, 82.5, 83.6, 85.1, 86.0, 86.8, 87.6, 88.2, 88.6, 89.2, 89.5]
sim_data = [65.5, 81.7, 83.6, 84.6, 85.4, 86.2, 86.9, 87.6, 88.0, 88.6, 88.9]
sim_std = [21.9, 19.8, 18.6, 16.4, 15.0, 14.1, 13.4, 13.0, 12.5, 11.9, 11.6]
sim_lower_bound = [M_new - Sigma/10 for M_new, Sigma in zip(sim_data, sim_std)]
sim_upper_bound = [M_new + Sigma/10 for M_new, Sigma in zip(sim_data, sim_std)]
# sim_conf = [95.2, 97.1, 96.8, 96.6, 96.0, 95.9, 95.8, 95.7, 95.6, 95.6]

# picle_data = [65.5, 83.1, 86.0, 88.9, 90.0, 90.7, 91.2, 91.7, 91.8, 92.3, 92.2]
picle_data = [65.5, 82.7, 86.2, 88.1, 90.0, 90.5, 91.1, 91.7, 91.8, 92.2, 92.3]
picle_std = [21.9, 20.9, 20.0, 17.5, 17.3, 17.0, 16.5, 16.5, 16.0, 15.4, 15.3]
lower_bound = [M_new - Sigma/10 for M_new, Sigma in zip(picle_data, picle_std)]
upper_bound = [M_new + Sigma/10 for M_new, Sigma in zip(picle_data, picle_std)]

plt.figure(figsize=(17,10))
sns.set_style('darkgrid')

sns.lineplot(picle_data[1:],linewidth=5,marker='o',markersize=15, label='PICLe')
sns.lineplot(sim_data[1:],linewidth=5,marker='^',markersize=15, label='Similarity')
# plt.fill_between(range(11), sim_lower_bound, sim_upper_bound, alpha=.3, color='blue')
# plt.fill_between(range(11), lower_bound, upper_bound, alpha=.3, color='red')

plt.legend(loc='lower right',fontsize='30')
plt.savefig('out/numex.png')
plt.close()