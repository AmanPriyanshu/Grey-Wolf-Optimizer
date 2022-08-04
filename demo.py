import numpy as np
from optimizer import GWO
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
from PIL import Image, ImageFile
from tqdm import tqdm
#ImageFile.LOAD_TRUNCATED_IMAGES = True

def fitness_function(X):
	x1 = X.T[0]
	x2 = X.T[1]
	return np.square(x1) - x1*x2 + np.square(x2) + 2*x1 + 4*x2 + 3

def search_space_assertion(x_arr, limits):
	asserted_x = []
	for limit, x in zip(limits, x_arr.T):
		tmp_x = np.where(x>=limit[0], x, limit[0])
		tmp_x = np.where(tmp_x<=limit[1], tmp_x, limit[1])
		asserted_x.append(tmp_x)
	return np.stack(asserted_x).T

def plot(intermediate_populations, gwo, base_N=250, tmp_name="tmp.jpg"):
	complete_pop = 10*(np.array([[[i, j] for j in range(base_N)] for i in range(base_N)])/base_N) - 5
	complete_pop = np.reshape(complete_pop, (-1, 2))
	losses = fitness_function(complete_pop)
	images = []
	width = None
	for pop in tqdm(intermediate_populations, desc="Plotting Graphs"):
		plt.cla()
		plt.clf()
		plt.rcParams["figure.figsize"] = (5,5)
		fig = plt.figure()
		ax = plt.axes(projection='3d')
		ax.plot_trisurf(complete_pop.T[0], complete_pop.T[1], losses, cmap='viridis', edgecolor='none', alpha=0.75)
		losses_pop = fitness_function(pop)
		x_alpha, x_beta, x_delta = gwo.get_hierarchy(pop, losses_pop)
		ax.scatter3D(pop.T[0], pop.T[1], losses_pop, c=losses_pop, cmap='Reds', edgecolor="y", label="Omega Wolf")
		ax.scatter3D(x_alpha[0:1], x_alpha[1:2], fitness_function(np.array([x_alpha])), c=fitness_function(np.array([x_alpha])), cmap="Reds", edgecolor="r", s=50, label="Alpha Wolf")
		ax.scatter3D(x_beta[0:1], x_beta[1:2], fitness_function(np.array([x_beta])), c=fitness_function(np.array([x_beta])), cmap="Reds", edgecolor="b", s=50, label="Beta Wolf")
		ax.scatter3D(x_delta[0:1], x_delta[1:2], fitness_function(np.array([x_delta])), c=fitness_function(np.array([x_delta])), cmap="Reds", edgecolor="m", s=50, label="Delta Wolf")
		plt.legend()
		plt.savefig(tmp_name)
		img = Image.open(tmp_name)
		if width is None:
			width, height = img.size
		else:
			img = img.resize((width, height))
		images.append(img)
	images[0].save('optimzation.gif',save_all=True, append_images=images[1:], optimize=False, duration=10, loop=0)

if __name__ == '__main__':
	print("Health Testing Fitness Function: (0, 0) =", fitness_function(np.array([[0, 0]])), "; and (0, 1) & (1, 0) = ", fitness_function(np.array([np.arange(2), np.arange(2)[::-1]])))
	print("Assertion Testing: (-6,... 6) & (6,... -6) = ", search_space_assertion(np.array([np.arange(-6, 7), np.arange(-6, 7)[::-1]]).T, ([-5, 5], [-5, 5])))
	gwo = GWO(50, fitness_function, search_space_assertion, limits=([-5, 5], [-5, 5]))
	pop = np.random.uniform(low=0, high=5, size=(100, 2))
	pop, x_alpha, x_beta, x_delta, intermediate_populations = gwo.search(pop, True)
	plot(intermediate_populations, gwo)