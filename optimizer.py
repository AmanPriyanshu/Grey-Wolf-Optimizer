import numpy as np
from tqdm import trange

class GWO:
	def __init__(self, num_iterations, fitness_function, search_space_assertion, limits):
		self.limits = limits
		self.fitness_function = fitness_function
		self.search_space_assertion = search_space_assertion
		self.num_iterations = num_iterations
		self.a_step = 2/(self.num_iterations-1)
		self.a = 2

	def get_hierarchy(self, population, losses):
		sorted_order = np.argsort(losses)
		x_alpha = population[sorted_order[0]]
		x_beta = population[sorted_order[1]]
		x_delta = population[sorted_order[2]]
		return x_alpha, x_beta, x_delta

	def encircle_prey(self, population, x_alpha, x_beta, x_delta):
		A = 2*self.a*np.random.random(size=(population.shape[0], 1)) - self.a
		C = 2*np.random.random(size=(population.shape[0], 1))
		D_alpha = np.abs(C * x_alpha - population)
		X_alpha = x_alpha - A * D_alpha
		D_beta = np.abs(C * x_beta - population)
		X_beta = x_beta - A * D_beta
		D_delta = np.abs(C * x_delta - population)
		X_delta = x_delta - A * D_delta
		return X_alpha, X_beta, X_delta

	def hunt(self, population, losses, X_alpha, X_beta, X_delta):
		X = (X_alpha + X_beta + X_delta)/3
		new_losses = self.fitness_function(X)
		updated_x_bool = np.where(new_losses<losses, 1, 0)
		new_population = []
		for i, b in enumerate(updated_x_bool):
			if b:
				new_population.append(X[i])
			else:
				new_population.append(population[i])
		new_population = np.stack(new_population)
		new_population = self.search_space_assertion(new_population, self.limits)
		return new_population

	def search(self, pop, return_intermediate_populations=False):
		bar = trange(1, self.num_iterations+1)
		if return_intermediate_populations:
			intermediate_populations = [pop]
		for iteration in bar:
			losses = self.fitness_function(pop)
			x_alpha, x_beta, x_delta = self.get_hierarchy(pop, losses)
			X_alpha, X_beta, X_delta = self.encircle_prey(pop, x_alpha, x_beta, x_delta)
			pop = self.hunt(pop, losses, X_alpha, X_beta, X_delta)
			bar.set_description(str({"iteration_no": str(iteration)+"/"+str(self.num_iterations), "loss": round(np.mean(losses), 3)}))
			if return_intermediate_populations:
				intermediate_populations.append(pop)
		bar.close()
		losses = self.fitness_function(pop)
		x_alpha, x_beta, x_delta = self.get_hierarchy(pop, losses)
		if return_intermediate_populations:
			return pop, x_alpha, x_beta, x_delta, intermediate_populations
		else:
			return pop, x_alpha, x_beta, x_delta