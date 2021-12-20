import numpy as np
from itertools import chain



class Genome:
    """
    A genome is represented by a unique id number
    and points to its parent. A mutation is
    added at uniform random position
    within the genome.
    """
    def __init__(self, genome_id, genome_size, parent=-1):
        self.genome_id = genome_id
        self.parent = parent
        self.mutation_position = np.random.randint(0, genome_size)
    def __str__(self):
        return (self.genome_id, self.parent, self.mutation_position)


class Population:
    def __init__(self, pop, N, mu, genome_size):
        """
        Population 
        pop (dict) mapping of a genome id to
        the number of individuals in the population with that
        genome. {0: 10000}
        N (int): constant size of population (if pop dict does not
        match this size, the first generations will double until 2N is
        reached)
        mu (float) the per genome per generation mutation rate.
        genome_size (int) Number of base_pairs in genome.
        """
        self.mu = mu
        self.pop = pop
        self.N = int(N)
        self.genome_size = int(genome_size)
        self.genomes = [Genome(i, self.genome_size) for i in set(self.pop.keys())]        
        # infinite allele model, start with one plus the max value in the pop dict.
        self.next_allele = max(map(int, pop.keys())) + 1
        
        
    def reproduce(self):
        """
        Make a copy of all genomes to represent
        doubling through cell division
        """
        self.pop = {k: v*2 for k, v in self.pop.items()}
    
    def mutate(self):
        """
        Add any mutations to next generation.
        """
        # total number of individuals in 2N population (both copies can gain mutations)
        N = sum(list(self.pop.values()))
        # Calculate the average number of mutations per generation
        avg_muts_per_gen = np.ceil(self.mu * N * self.genome_size)
        # Draw the number of mutations that will occur this generation
        # max of every genome receives a mutation
        n_mutations = min(N, np.random.poisson(avg_muts_per_gen))
        # Make exanded array of all alleles for sampling without replacement
        alleles = list(
                chain.from_iterable(
                    np.array([np.repeat(k, v) for k, v in self.pop.items()])))
        # pick random genome ids to be mutated.
        mutate_genomes = np.random.choice(
            alleles,
            size=n_mutations,
            replace=False)
        # Reduce the mutated genome counts in pop and add new genome ids
        # Note: assumes that the same genome can't be mutated twice in a generation
        # this is a good assumption assuming that the mutation rate is low and populations
        # are large.
        for genome in mutate_genomes:
            # remove count from previous genome id
            self.pop[genome] -= 1
            # create a new genome id with a count of 1
            self.pop[self.next_allele] = 1
            # Add a new Genome object pointing at parent genome
            self.genomes.append(Genome(self.next_allele, self.genome_size, genome))
            # Increment next allele number
            self.next_allele += 1
            
    def selection(self):
        """
        Sample without replacement from doubled population down to N individuals.
        """
        # get array of all alleles in current population
        N = sum(list(self.pop.values()))
        # only perform selection if the population is over the constant size
        if N > self.N:
            alleles = list(
                chain.from_iterable(
                    np.array([np.repeat(k, v) for k, v in self.pop.items()])))
            self.pop = {k: v for k, v in
                zip(*np.unique(np.random.choice(
                alleles,
                size=self.N,
                replace=False), return_counts=True))}

    def evolve(self):
        """
        Run a single generation step of reproduction,
        mutation, and selection
        """
        self.reproduce()
        self.mutate()
        self.selection()
    
    def sample(self, N, replace=False):
        """
        Return a sample from the population in pop dict format
        """
        alleles = list(
            chain.from_iterable(
                np.array([np.repeat(k, v) for k, v in self.pop.items()])))
        return {k: v for k, v in
            zip(*np.unique(np.random.choice(
            alleles,
            size=N,
            replace=replace), return_counts=True))}
        
        

    