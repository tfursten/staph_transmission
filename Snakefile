
import json
import pandas as pd
import numpy as np
from collections import defaultdict
from src.Population import Population
from itertools import chain


def run_sim(population, n_generations):
    """
    Run simulation on a population for n_generations
    """
    for gen in range(n_generations):
        if gen%500 == 0:
            print(gen)
        population.evolve()
    return population

def json_keys2int(x):
    if isinstance(x, dict):
            return {int(k):v for k,v in x.items()}
    return x

def get_population_from_json(json_file):
    return json.loads(open(json_file, 'r').read(), object_hook=json_keys2int)

def create_population(pop, pop_size, mutation_rate, genome_size):
    return Population(
        pop, int(pop_size), float(mutation_rate), int(genome_size))

def write_lineage_info(outfile, pop):
    with open(outfile, 'w') as out:
        out.write("GenomeID,ParentID,GenomePos\n")
        for genome in pop.genomes:
            out.write("{0},{1},{2}\n".format(
                genome.genome_id, genome.parent,
                genome.mutation_position))

def write_genome_info(outfile, pop):
    with open(outfile, 'w') as out:
        out.write("GenomeID,Count\n")
        for k, v in pop.pop.items():
            out.write("{0},{1}\n".format(k, v))

def sample_from_population(count_data, sample_size):
    """
    Sample from population, if sample_size is not provided
    returns the full population.
    """
    counts = count_data['Count']
    genome_ids = count_data['GenomeID']
    total = counts.sum()
    if sample_size == None:
        sample_size = total
    replacement = False
    if total <= sample_size:
        print("Warning sample size is larger than population size, sampling with replacement")
        replacment = True
    alleles = list(
            chain.from_iterable(
                np.array([np.repeat(k, v) for k, v in zip(genome_ids, counts)])))
    return {str(k): int(v) for k, v in
            zip(*np.unique(np.random.choice(
            alleles,
            size=sample_size,
            replace=replacement), return_counts=True))}



def write_sample_from_population(infile, outfile, sample_size=None):
    df = pd.read_csv(infile)
    selected_genomes = sample_from_population(df, sample_size)
    with open(outfile, 'w') as out:
        out.write(json.dumps(selected_genomes))


def lineage_traceback(lineage_df, sample, output):
    """
    Follow lineage of each genome in the sample
    and return the genotype of each genome.
    """
    genotype_data = {}
    # keep track of last ancestor to link up with
    # older lineages
    founders = {}
    for genome, count in sample.items():
        index = int(genome)
        genome_dict = {}
        
        while index != -1:
            row = lineage_df.loc[index]
            parent = int(row['ParentID'])
            pos = int(row['GenomePos'])
            if parent == -1:
                # Done with traceback, mark original founder genome
                # to link with previous lineages
                founders[genome] = int(row.name)
            index = parent
            # Increment number of mutations at genome location
            if pos in genome_dict:
                genome_dict[pos] += 1
            else:
                genome_dict[pos] = 1
        genotype_data[genome] = genome_dict
        
    return {'genotypes': genotype_data, 'founders': founders}


def write_lineage_traceback(lineage, sample, outfile):
    sample = json.loads(open(sample, 'r').read())
    lineage_df = pd.read_csv(lineage, index_col=0, dtype=int)
    genotypes = lineage_traceback(lineage_df, sample, outfile)
    with open(outfile, 'w') as out:
        out.write(json.dumps(genotypes))


def add_lineage_genotypes(lineage_1, lineage_2):
    """
    Combine two lineages by summing the number of mutations at
    each genome location. 
    """
    for k, v in lineage_2.items():
        if k in lineage_1:
            lineage_1[k] += 1
        else:
            lineage_1[k] = 1
    return lineage_1


def combine_lineages(burnin, source, transmission):
    combined_lineages = {}
    for genome, mutations in transmission['genotypes'].items():
        source_founder = transmission['founders'][str(genome)]
        burnin_founder = source['founders'][str(source_founder)]
        source_lineage = add_lineage_genotypes(mutations, source['genotypes'][str(source_founder)])
        burnin_lineage = add_lineage_genotypes(source_lineage, burnin['genotypes'][str(burnin_founder)])
        combined_lineages[genome] = burnin_lineage
    return combined_lineages
    

def write_combined_lineages(burnin_traceback, source_traceback, transmission_traceback, output):
    burnin = json.loads(open(burnin_traceback, 'r').read())
    source = json.loads(open(source_traceback, 'r').read())
    transmission = json.loads(open(transmission_traceback, 'r').read())
    lineages = combine_lineages(burnin, source, transmission)
    with open(output, 'w') as out:
        out.write(json.dumps(lineages))


def get_unique_positions(genotypes):
    unique_positions = set()
    for genome, snps in genotypes.items():
        unique_positions.update(snps.keys())
    return unique_positions


def snp_matrix(genomes, genotypes):
    unique_positions = list(get_unique_positions(genotypes))

    snps = pd.DataFrame(
        np.zeros((len(genomes), len(unique_positions))),
        columns=unique_positions, index=list(genomes.keys()))
    for genome, genotypes in genotypes.items():
        for pos, snp in genotypes.items():
            snps.loc[genome, pos] = snp
    snps['counts'] = [genomes[idx] for idx in snps.index]
    return snps

def write_snp_matrix(genomes, lineage, outfile):
    genomes = json.loads(open(genomes, 'r').read())
    genotypes = json.loads(open(lineage, 'r').read())
    snps = snp_matrix(genomes, genotypes)
    snps.to_csv(outfile)

########## PARAMS #########

# 	• Scaling of certain population parameters can be done to reduce the time of the burn-in.
# 		- Theta (genetic diversity) = 2x Ne x Mu (mutation rate).
# 		- We can decrease the effective population size (Ne) 
#           by a factor of 10 while increasing the mutation rate by the same factor.
#           This will then reduce the length of the simulation by a factor of 10.

# - A colonized nasal cavity can carry 10^4-10^8 cells
# - in vivo doubling time (generation time) is around 1-4 hours
# - The mutation rate is between 1.6 - 3.4x10-10 mutations per nucleotide per generation

scaling_factor = 50
# pop size for burnin pop
burnin_size = [int(1000000/scaling_factor)]
# mutation rate during burnin
burnin_mu = [0.00000000016 * scaling_factor]
# number of generations for burnin (typically 5 x scaled population size but probably can be smaller)
burnin_gen = [5000]
# genome size
genome_size = [2800000]
# Sample replicate IDs for multiple draws from burnin pop
burnin_sample_rep = [1]
# Number of cells transferred from burnin pop to source pop
burnin_sample_size = [1]
# Max size of source population
source_size = [int(1000000/scaling_factor)]
# Mutation rate of source population
source_mu = [0.00000000016 * scaling_factor]
# number of generations to run source pop prior to transmission event
source_gen = [1000]
# Sample replicate IDs for multiple draws from source pop
source_sample_rep = [1,2,3]
# Number of cells transferred from source population during transmission
source_sample_size = [1]
# Max size of transmission population
transmission_size = [int(1000000/scaling_factor)]
# Mutation rate in transmission population (and post-transmission source population)
transmission_mu = [0.00000000016 * scaling_factor]
# Number of generations to run transmission population and post-transmission source population
transmission_gen = [100]
# Sample replicate IDs for multiple collections from final transmission population
trans_collection_sample_rep = [1, 2, 3]
# Size of sample from final transmission population (represents swab and colony selection)
trans_collection_sample_size = [30]
# Sample replicate IDs for multiple collections from final post-transmission source population
source_collection_sample_rep = [1, 2, 3]
# Size of sample from final post-transmission source population (represents swab and colony selection)
source_collection_sample_size = [30]

rule all:
    input: 
        expand(
            "sample_collection/transmission_sample_snp_matrix_bsz_{burnin_size}_bmu_{burnin_mu}_bgen_{burnin_gen}_g_{genome_size}_brep_{burnin_sample_rep}_bssz_{burnin_sample_size}_ssz_{source_size}_smu_{source_mu}_sgen_{source_gen}_srep_{source_sample_rep}_sssz_{source_sample_size}_tsz_{transmission_size}_tmu_{transmission_mu}_tgen_{transmission_gen}_tcrep_{trans_collection_sample_rep}_tcssz_{trans_collection_sample_size}.csv",
            burnin_size=burnin_size,
            burnin_mu=burnin_mu,
            burnin_gen=burnin_gen,
            genome_size=genome_size,
            burnin_sample_rep=burnin_sample_rep,
            burnin_sample_size=burnin_sample_size,
            source_size=source_size,
            source_mu=source_mu,
            source_gen=source_gen,
            source_sample_rep=source_sample_rep,
            source_sample_size=source_sample_size,
            transmission_size=transmission_size,
            transmission_mu=transmission_mu,
            transmission_gen=transmission_gen,
            trans_collection_sample_rep=trans_collection_sample_rep,
            trans_collection_sample_size=trans_collection_sample_size),
        expand(
            "sample_collection/source_sample_snp_matrix_bsz_{burnin_size}_bmu_{burnin_mu}_bgen_{burnin_gen}_g_{genome_size}_brep_{burnin_sample_rep}_bssz_{burnin_sample_size}_ssz_{source_size}_smu_{source_mu}_sgen_{source_gen}_srep_{source_sample_rep}_sssz_{source_sample_size}_tsz_{transmission_size}_tmu_{transmission_mu}_tgen_{transmission_gen}_screp_{source_collection_sample_rep}_scssz_{source_collection_sample_size}.csv",
            burnin_size=burnin_size,
            burnin_mu=burnin_mu,
            burnin_gen=burnin_gen,
            genome_size=genome_size,
            burnin_sample_rep=burnin_sample_rep,
            burnin_sample_size=burnin_sample_size,
            source_size=source_size,
            source_mu=source_mu,
            source_gen=source_gen,
            source_sample_rep=source_sample_rep,
            source_sample_size=source_sample_size,
            transmission_size=transmission_size,
            transmission_mu=transmission_mu,
            transmission_gen=transmission_gen,
            source_collection_sample_rep=source_collection_sample_rep,
            source_collection_sample_size=source_collection_sample_size)
    

rule burnin:
    """
    Simulate a burn-in period to reach drift/mutation equilibrium and simulate deeper lineages.
    """
    output:
        lineage="burnin/burnin_lineage_bsz_{burnin_size}_bmu_{burnin_mu}_bgen_{burnin_gen}_g_{genome_size}.txt",
        genomes="burnin/burnin_genomes_bsz_{burnin_size}_bmu_{burnin_mu}_bgen_{burnin_gen}_g_{genome_size}.txt"
    params:
        pop_size = lambda wc: int(wc.burnin_size),
        mutation_rate = lambda wc: float(wc.burnin_mu),
        n_generations = lambda wc: int(wc.burnin_gen),
        genome_size = lambda wc: int(wc.genome_size)
    run:
        # initialize population with all same starting genome
        init_pop = {0: int(params.pop_size)}
        init_pop = create_population(
            init_pop, params.pop_size, params.mutation_rate, params.genome_size)
        burnin_pop = run_sim(init_pop, params.n_generations)
        write_genome_info(output.genomes, burnin_pop)
        write_lineage_info(output.lineage, burnin_pop)


rule sample_from_burnin:
    """
    Samples genomes from burnin and writes pop to json file.
    This population will be used to initialize the source
    population during the transmission event.
    """
    input: "burnin/burnin_genomes_bsz_{burnin_size}_bmu_{burnin_mu}_bgen_{burnin_gen}_g_{genome_size}.txt"
    output: "burnin/burnin_simulation_sample_bsz_{burnin_size}_bmu_{burnin_mu}_bgen_{burnin_gen}_g_{genome_size}_brep_{burnin_sample_rep}_bssz_{burnin_sample_size}.json"
    params:
        sample_size = lambda wc: int(wc.burnin_sample_size)
    run:
        write_sample_from_population(input[0], output[0], params.sample_size)


rule trace_burnin_sample_lineage:
    """
    Get burnin sample genotypes by tracing back lineage
    """
    input:
        lineage = "burnin/burnin_lineage_bsz_{burnin_size}_bmu_{burnin_mu}_bgen_{burnin_gen}_g_{genome_size}.txt",
        sample = "burnin/burnin_simulation_sample_bsz_{burnin_size}_bmu_{burnin_mu}_bgen_{burnin_gen}_g_{genome_size}_brep_{burnin_sample_rep}_bssz_{burnin_sample_size}.json"
    output: "burnin/burnin_sample_genotypes_bsz_{burnin_size}_bmu_{burnin_mu}_bgen_{burnin_gen}_g_{genome_size}_brep_{burnin_sample_rep}_bssz_{burnin_sample_size}.json"
    run:
        write_lineage_traceback(input.lineage, input.sample, output[0])


rule evolve_source_population:
    input:
        init_pop="burnin/burnin_simulation_sample_bsz_{burnin_size}_bmu_{burnin_mu}_bgen_{burnin_gen}_g_{genome_size}_brep_{burnin_sample_rep}_bssz_{burnin_sample_size}.json"
    output:
        lineage="source/source_lineage_bsz_{burnin_size}_bmu_{burnin_mu}_bgen_{burnin_gen}_g_{genome_size}_brep_{burnin_sample_rep}_bssz_{burnin_sample_size}_ssz_{source_size}_smu_{source_mu}_sgen_{source_gen}.txt",
        genomes="source/source_genomes_bsz_{burnin_size}_bmu_{burnin_mu}_bgen_{burnin_gen}_g_{genome_size}_brep_{burnin_sample_rep}_bssz_{burnin_sample_size}_ssz_{source_size}_smu_{source_mu}_sgen_{source_gen}.txt",
    params:
        pop_size = lambda wc: int(wc.source_size),
        mutation_rate = lambda wc: float(wc.source_mu),
        n_generations = lambda wc: int(wc.source_gen),
        genome_size = lambda wc: int(wc.genome_size)
    run:
        # initialize population with sample from burnin
        init_pop = get_population_from_json(input.init_pop)
        init_pop = create_population(
            init_pop, params.pop_size, params.mutation_rate, params.genome_size)
        source_pop = run_sim(init_pop, params.n_generations)
        write_genome_info(output.genomes, source_pop)
        write_lineage_info(output.lineage, source_pop)
        

rule sample_from_source:
    """
    Samples genomes from source population and writes pop to json file.
    This population will represent the transmission event and therefore
    the size of the sample is the transmission bottleneck size.
    """
    input: "source/source_genomes_bsz_{burnin_size}_bmu_{burnin_mu}_bgen_{burnin_gen}_g_{genome_size}_brep_{burnin_sample_rep}_bssz_{burnin_sample_size}_ssz_{source_size}_smu_{source_mu}_sgen_{source_gen}.txt"
    output: "source/source_simulation_sample_bsz_{burnin_size}_bmu_{burnin_mu}_bgen_{burnin_gen}_g_{genome_size}_brep_{burnin_sample_rep}_bssz_{burnin_sample_size}_ssz_{source_size}_smu_{source_mu}_sgen_{source_gen}_srep_{source_sample_rep}_sssz_{source_sample_size}.json"
    params:
        sample_size = lambda wc: int(wc.source_sample_size)
    run:
        write_sample_from_population(input[0], output[0], params.sample_size)


rule trace_source_sample_lineage:
    """
    Get source sample genotypes by tracing back lineage
    """
    input:
        lineage = "source/source_lineage_bsz_{burnin_size}_bmu_{burnin_mu}_bgen_{burnin_gen}_g_{genome_size}_brep_{burnin_sample_rep}_bssz_{burnin_sample_size}_ssz_{source_size}_smu_{source_mu}_sgen_{source_gen}.txt",
        sample = "source/source_simulation_sample_bsz_{burnin_size}_bmu_{burnin_mu}_bgen_{burnin_gen}_g_{genome_size}_brep_{burnin_sample_rep}_bssz_{burnin_sample_size}_ssz_{source_size}_smu_{source_mu}_sgen_{source_gen}_srep_{source_sample_rep}_sssz_{source_sample_size}.json"
    output: "source/source_sample_genotypes_bsz_{burnin_size}_bmu_{burnin_mu}_bgen_{burnin_gen}_g_{genome_size}_brep_{burnin_sample_rep}_bssz_{burnin_sample_size}_ssz_{source_size}_smu_{source_mu}_sgen_{source_gen}_srep_{source_sample_rep}_sssz_{source_sample_size}.json"
    run:
        write_lineage_traceback(input.lineage, input.sample, output[0])


rule continue_source:
    """
    Sample complete source population to continue source population after transmission
    """
    input: "source/source_genomes_bsz_{burnin_size}_bmu_{burnin_mu}_bgen_{burnin_gen}_g_{genome_size}_brep_{burnin_sample_rep}_bssz_{burnin_sample_size}_ssz_{source_size}_smu_{source_mu}_sgen_{source_gen}.txt"
    output: "source/source_simulation_continue_bsz_{burnin_size}_bmu_{burnin_mu}_bgen_{burnin_gen}_g_{genome_size}_brep_{burnin_sample_rep}_bssz_{burnin_sample_size}_ssz_{source_size}_smu_{source_mu}_sgen_{source_gen}_srep_{source_sample_rep}_sssz_{source_sample_size}.json"
    run:
        write_sample_from_population(input[0], output[0])

rule trace_complete_source_population:
    """
    Get complete source genotypes by tracing back lineage
    """
    input:
        lineage = "source/source_lineage_bsz_{burnin_size}_bmu_{burnin_mu}_bgen_{burnin_gen}_g_{genome_size}_brep_{burnin_sample_rep}_bssz_{burnin_sample_size}_ssz_{source_size}_smu_{source_mu}_sgen_{source_gen}.txt",
        sample = "source/source_simulation_continue_bsz_{burnin_size}_bmu_{burnin_mu}_bgen_{burnin_gen}_g_{genome_size}_brep_{burnin_sample_rep}_bssz_{burnin_sample_size}_ssz_{source_size}_smu_{source_mu}_sgen_{source_gen}_srep_{source_sample_rep}_sssz_{source_sample_size}.json"
    output: "source/complete_source_genotypes_bsz_{burnin_size}_bmu_{burnin_mu}_bgen_{burnin_gen}_g_{genome_size}_brep_{burnin_sample_rep}_bssz_{burnin_sample_size}_ssz_{source_size}_smu_{source_mu}_sgen_{source_gen}_srep_{source_sample_rep}_sssz_{source_sample_size}.json"
    run:
        write_lineage_traceback(input.lineage, input.sample, output[0])



rule simulate_transmission_population:
    """
    Simulate generations of the transmission population. The population is initialized with a sample
    from the source population. 
    """
    input:
        init_pop="source/source_simulation_sample_bsz_{burnin_size}_bmu_{burnin_mu}_bgen_{burnin_gen}_g_{genome_size}_brep_{burnin_sample_rep}_bssz_{burnin_sample_size}_ssz_{source_size}_smu_{source_mu}_sgen_{source_gen}_srep_{source_sample_rep}_sssz_{source_sample_size}.json",
    output:
        lineage="transmission/transmission_lineage_bsz_{burnin_size}_bmu_{burnin_mu}_bgen_{burnin_gen}_g_{genome_size}_brep_{burnin_sample_rep}_bssz_{burnin_sample_size}_ssz_{source_size}_smu_{source_mu}_sgen_{source_gen}_srep_{source_sample_rep}_sssz_{source_sample_size}_tsz_{transmission_size}_tmu_{transmission_mu}_tgen_{transmission_gen}.txt",
        genomes="transmission/transmission_genomes_bsz_{burnin_size}_bmu_{burnin_mu}_bgen_{burnin_gen}_g_{genome_size}_brep_{burnin_sample_rep}_bssz_{burnin_sample_size}_ssz_{source_size}_smu_{source_mu}_sgen_{source_gen}_srep_{source_sample_rep}_sssz_{source_sample_size}_tsz_{transmission_size}_tmu_{transmission_mu}_tgen_{transmission_gen}.txt",
    params:
        pop_size = lambda wc: int(wc.transmission_size),
        mutation_rate = lambda wc: float(wc.transmission_mu),
        n_generations = lambda wc: int(wc.transmission_gen),
        genome_size = lambda wc: int(wc.genome_size)
    run:
        # initialize transmission population with sample from source population
        init_pop = get_population_from_json(input.init_pop)
        init_pop = create_population(
            init_pop, params.pop_size, params.mutation_rate, params.genome_size)
        transmission_pop = run_sim(init_pop, params.n_generations)
        write_genome_info(output.genomes, transmission_pop)
        write_lineage_info(output.lineage, transmission_pop)

rule simulate_source_post_transmission:
    """
    Continue simulating the source population for same time as transmission population.
    Uses the same mutation rate and number of generations as transmission population
    but the same pop size as original source population. Simulation picks up where 
    source left off.
    """
    input:
        genomes="source/source_genomes_bsz_{burnin_size}_bmu_{burnin_mu}_bgen_{burnin_gen}_g_{genome_size}_brep_{burnin_sample_rep}_bssz_{burnin_sample_size}_ssz_{source_size}_smu_{source_mu}_sgen_{source_gen}.txt"
    output:
        lineage="transmission/source_lineage_bsz_{burnin_size}_bmu_{burnin_mu}_bgen_{burnin_gen}_g_{genome_size}_brep_{burnin_sample_rep}_bssz_{burnin_sample_size}_ssz_{source_size}_smu_{source_mu}_sgen_{source_gen}_srep_{source_sample_rep}_sssz_{source_sample_size}_tsz_{transmission_size}_tmu_{transmission_mu}_tgen_{transmission_gen}.txt",
        genomes="transmission/source_genomes_bsz_{burnin_size}_bmu_{burnin_mu}_bgen_{burnin_gen}_g_{genome_size}_brep_{burnin_sample_rep}_bssz_{burnin_sample_size}_ssz_{source_size}_smu_{source_mu}_sgen_{source_gen}_srep_{source_sample_rep}_sssz_{source_sample_size}_tsz_{transmission_size}_tmu_{transmission_mu}_tgen_{transmission_gen}.txt",
    params:
        pop_size = lambda wc: int(wc.source_size),
        mutation_rate = lambda wc: float(wc.transmission_mu),
        n_generations = lambda wc: int(wc.transmission_gen),
        genome_size = lambda wc: int(wc.genome_size)
    run:
        init_pop = pd.read_csv(input.genomes)
        init_pop = {k: v for _, (k, v) in init_pop.iterrows()}
        init_pop = create_population(
            init_pop, params.pop_size, params.mutation_rate, params.genome_size)
        source_pop = run_sim(init_pop, params.n_generations)
        write_genome_info(output.genomes, source_pop)
        write_lineage_info(output.lineage, source_pop)



rule sample_from_transmission_population:
    """
    Collect a sample from the transmission population.
    This represents the sample size after swabbing
    and culturing and selecting colonies.
    """
    input: "transmission/transmission_genomes_bsz_{burnin_size}_bmu_{burnin_mu}_bgen_{burnin_gen}_g_{genome_size}_brep_{burnin_sample_rep}_bssz_{burnin_sample_size}_ssz_{source_size}_smu_{source_mu}_sgen_{source_gen}_srep_{source_sample_rep}_sssz_{source_sample_size}_tsz_{transmission_size}_tmu_{transmission_mu}_tgen_{transmission_gen}.txt",
    output: "sample_collection/transmission_sample_bsz_{burnin_size}_bmu_{burnin_mu}_bgen_{burnin_gen}_g_{genome_size}_brep_{burnin_sample_rep}_bssz_{burnin_sample_size}_ssz_{source_size}_smu_{source_mu}_sgen_{source_gen}_srep_{source_sample_rep}_sssz_{source_sample_size}_tsz_{transmission_size}_tmu_{transmission_mu}_tgen_{transmission_gen}_tcrep_{trans_collection_sample_rep}_tcssz_{trans_collection_sample_size}.json"
    params:
        sample_size = lambda wc: int(wc.trans_collection_sample_size)
    run:
        write_sample_from_population(input[0], output[0], params.sample_size)
    
    
rule sample_from_post_transmission_source_population:
    """
    Collect a sample from the post transmission source population.
    This represents the sample size after swabbing
    and culturing and selecting colonies.
    """
    input: "transmission/source_genomes_bsz_{burnin_size}_bmu_{burnin_mu}_bgen_{burnin_gen}_g_{genome_size}_brep_{burnin_sample_rep}_bssz_{burnin_sample_size}_ssz_{source_size}_smu_{source_mu}_sgen_{source_gen}_srep_{source_sample_rep}_sssz_{source_sample_size}_tsz_{transmission_size}_tmu_{transmission_mu}_tgen_{transmission_gen}.txt",
    output: "sample_collection/source_sample_bsz_{burnin_size}_bmu_{burnin_mu}_bgen_{burnin_gen}_g_{genome_size}_brep_{burnin_sample_rep}_bssz_{burnin_sample_size}_ssz_{source_size}_smu_{source_mu}_sgen_{source_gen}_srep_{source_sample_rep}_sssz_{source_sample_size}_tsz_{transmission_size}_tmu_{transmission_mu}_tgen_{transmission_gen}_screp_{source_collection_sample_rep}_scssz_{source_collection_sample_size}.json"
    params:
        sample_size = lambda wc: int(wc.source_collection_sample_size)
    run:
        write_sample_from_population(input[0], output[0], params.sample_size)


rule trace_transmission_sample_lineage:
    """
    Get source sample genotypes by tracing back lineage
    """
    input:
        lineage = "transmission/transmission_lineage_bsz_{burnin_size}_bmu_{burnin_mu}_bgen_{burnin_gen}_g_{genome_size}_brep_{burnin_sample_rep}_bssz_{burnin_sample_size}_ssz_{source_size}_smu_{source_mu}_sgen_{source_gen}_srep_{source_sample_rep}_sssz_{source_sample_size}_tsz_{transmission_size}_tmu_{transmission_mu}_tgen_{transmission_gen}.txt",
        sample = "sample_collection/transmission_sample_bsz_{burnin_size}_bmu_{burnin_mu}_bgen_{burnin_gen}_g_{genome_size}_brep_{burnin_sample_rep}_bssz_{burnin_sample_size}_ssz_{source_size}_smu_{source_mu}_sgen_{source_gen}_srep_{source_sample_rep}_sssz_{source_sample_size}_tsz_{transmission_size}_tmu_{transmission_mu}_tgen_{transmission_gen}_tcrep_{trans_collection_sample_rep}_tcssz_{trans_collection_sample_size}.json"
    output: "sample_collection/transmission_sample_genotypes_bsz_{burnin_size}_bmu_{burnin_mu}_bgen_{burnin_gen}_g_{genome_size}_brep_{burnin_sample_rep}_bssz_{burnin_sample_size}_ssz_{source_size}_smu_{source_mu}_sgen_{source_gen}_srep_{source_sample_rep}_sssz_{source_sample_size}_tsz_{transmission_size}_tmu_{transmission_mu}_tgen_{transmission_gen}_tcrep_{trans_collection_sample_rep}_tcssz_{trans_collection_sample_size}.json" 
    run:
        write_lineage_traceback(input.lineage, input.sample, output[0])

rule trace_post_transmission_source_sample_lineage:
    """
    Get source sample genotypes by tracing back lineage
    """
    input:
        lineage = "transmission/source_lineage_bsz_{burnin_size}_bmu_{burnin_mu}_bgen_{burnin_gen}_g_{genome_size}_brep_{burnin_sample_rep}_bssz_{burnin_sample_size}_ssz_{source_size}_smu_{source_mu}_sgen_{source_gen}_srep_{source_sample_rep}_sssz_{source_sample_size}_tsz_{transmission_size}_tmu_{transmission_mu}_tgen_{transmission_gen}.txt",
        sample = "sample_collection/source_sample_bsz_{burnin_size}_bmu_{burnin_mu}_bgen_{burnin_gen}_g_{genome_size}_brep_{burnin_sample_rep}_bssz_{burnin_sample_size}_ssz_{source_size}_smu_{source_mu}_sgen_{source_gen}_srep_{source_sample_rep}_sssz_{source_sample_size}_tsz_{transmission_size}_tmu_{transmission_mu}_tgen_{transmission_gen}_screp_{source_collection_sample_rep}_scssz_{source_collection_sample_size}.json"
    output: "sample_collection/source_sample_genotypes_bsz_{burnin_size}_bmu_{burnin_mu}_bgen_{burnin_gen}_g_{genome_size}_brep_{burnin_sample_rep}_bssz_{burnin_sample_size}_ssz_{source_size}_smu_{source_mu}_sgen_{source_gen}_srep_{source_sample_rep}_sssz_{source_sample_size}_tsz_{transmission_size}_tmu_{transmission_mu}_tgen_{transmission_gen}_screp_{source_collection_sample_rep}_scssz_{source_collection_sample_size}.json"
    run:
        write_lineage_traceback(input.lineage, input.sample, output[0])
    
     
rule combine_transmission_lineages:
    """
    Combine the complete lineages across the burnin, source, and transmission samples
    """
    input:
        burnin_traceback = "burnin/burnin_sample_genotypes_bsz_{burnin_size}_bmu_{burnin_mu}_bgen_{burnin_gen}_g_{genome_size}_brep_{burnin_sample_rep}_bssz_{burnin_sample_size}.json",
        source_traceback = "source/source_sample_genotypes_bsz_{burnin_size}_bmu_{burnin_mu}_bgen_{burnin_gen}_g_{genome_size}_brep_{burnin_sample_rep}_bssz_{burnin_sample_size}_ssz_{source_size}_smu_{source_mu}_sgen_{source_gen}_srep_{source_sample_rep}_sssz_{source_sample_size}.json",
        transmission_traceback = "sample_collection/transmission_sample_genotypes_bsz_{burnin_size}_bmu_{burnin_mu}_bgen_{burnin_gen}_g_{genome_size}_brep_{burnin_sample_rep}_bssz_{burnin_sample_size}_ssz_{source_size}_smu_{source_mu}_sgen_{source_gen}_srep_{source_sample_rep}_sssz_{source_sample_size}_tsz_{transmission_size}_tmu_{transmission_mu}_tgen_{transmission_gen}_tcrep_{trans_collection_sample_rep}_tcssz_{trans_collection_sample_size}.json" 
    output: "sample_collection/transmission_complete_lineage_bsz_{burnin_size}_bmu_{burnin_mu}_bgen_{burnin_gen}_g_{genome_size}_brep_{burnin_sample_rep}_bssz_{burnin_sample_size}_ssz_{source_size}_smu_{source_mu}_sgen_{source_gen}_srep_{source_sample_rep}_sssz_{source_sample_size}_tsz_{transmission_size}_tmu_{transmission_mu}_tgen_{transmission_gen}_tcrep_{trans_collection_sample_rep}_tcssz_{trans_collection_sample_size}.json"
    run:
        write_combined_lineages(
            input.burnin_traceback, input.source_traceback, input.transmission_traceback, output[0])
        

rule combine_post_transmission_source_lineages:
    """
    Combine the complete lineages across the burnin, source, and and post transmission source samples
    """
    input:
        burnin_traceback = "burnin/burnin_sample_genotypes_bsz_{burnin_size}_bmu_{burnin_mu}_bgen_{burnin_gen}_g_{genome_size}_brep_{burnin_sample_rep}_bssz_{burnin_sample_size}.json",
        source_traceback = "source/complete_source_genotypes_bsz_{burnin_size}_bmu_{burnin_mu}_bgen_{burnin_gen}_g_{genome_size}_brep_{burnin_sample_rep}_bssz_{burnin_sample_size}_ssz_{source_size}_smu_{source_mu}_sgen_{source_gen}_srep_{source_sample_rep}_sssz_{source_sample_size}.json",
        transmission_traceback = "sample_collection/source_sample_genotypes_bsz_{burnin_size}_bmu_{burnin_mu}_bgen_{burnin_gen}_g_{genome_size}_brep_{burnin_sample_rep}_bssz_{burnin_sample_size}_ssz_{source_size}_smu_{source_mu}_sgen_{source_gen}_srep_{source_sample_rep}_sssz_{source_sample_size}_tsz_{transmission_size}_tmu_{transmission_mu}_tgen_{transmission_gen}_screp_{source_collection_sample_rep}_scssz_{source_collection_sample_size}.json"
    output: "sample_collection/source_complete_lineage_bsz_{burnin_size}_bmu_{burnin_mu}_bgen_{burnin_gen}_g_{genome_size}_brep_{burnin_sample_rep}_bssz_{burnin_sample_size}_ssz_{source_size}_smu_{source_mu}_sgen_{source_gen}_srep_{source_sample_rep}_sssz_{source_sample_size}_tsz_{transmission_size}_tmu_{transmission_mu}_tgen_{transmission_gen}_screp_{source_collection_sample_rep}_scssz_{source_collection_sample_size}.json"
    run:
        write_combined_lineages(
            input.burnin_traceback, input.source_traceback, input.transmission_traceback, output[0])



rule get_transmission_sample_snp_matrix:
    """
    create SNP matrix from transmission sample
    """
    input:
        lineage="sample_collection/transmission_complete_lineage_bsz_{burnin_size}_bmu_{burnin_mu}_bgen_{burnin_gen}_g_{genome_size}_brep_{burnin_sample_rep}_bssz_{burnin_sample_size}_ssz_{source_size}_smu_{source_mu}_sgen_{source_gen}_srep_{source_sample_rep}_sssz_{source_sample_size}_tsz_{transmission_size}_tmu_{transmission_mu}_tgen_{transmission_gen}_tcrep_{trans_collection_sample_rep}_tcssz_{trans_collection_sample_size}.json",
        sample="sample_collection/transmission_sample_bsz_{burnin_size}_bmu_{burnin_mu}_bgen_{burnin_gen}_g_{genome_size}_brep_{burnin_sample_rep}_bssz_{burnin_sample_size}_ssz_{source_size}_smu_{source_mu}_sgen_{source_gen}_srep_{source_sample_rep}_sssz_{source_sample_size}_tsz_{transmission_size}_tmu_{transmission_mu}_tgen_{transmission_gen}_tcrep_{trans_collection_sample_rep}_tcssz_{trans_collection_sample_size}.json" 
    output: "sample_collection/transmission_sample_snp_matrix_bsz_{burnin_size}_bmu_{burnin_mu}_bgen_{burnin_gen}_g_{genome_size}_brep_{burnin_sample_rep}_bssz_{burnin_sample_size}_ssz_{source_size}_smu_{source_mu}_sgen_{source_gen}_srep_{source_sample_rep}_sssz_{source_sample_size}_tsz_{transmission_size}_tmu_{transmission_mu}_tgen_{transmission_gen}_tcrep_{trans_collection_sample_rep}_tcssz_{trans_collection_sample_size}.csv" 
    run:
        write_snp_matrix(input.sample, input.lineage, output[0])


rule get_post_transmission_source_sample_snp_matrix:
    """
    Create SNP matrix from sample
    """
    input:
        lineage="sample_collection/source_complete_lineage_bsz_{burnin_size}_bmu_{burnin_mu}_bgen_{burnin_gen}_g_{genome_size}_brep_{burnin_sample_rep}_bssz_{burnin_sample_size}_ssz_{source_size}_smu_{source_mu}_sgen_{source_gen}_srep_{source_sample_rep}_sssz_{source_sample_size}_tsz_{transmission_size}_tmu_{transmission_mu}_tgen_{transmission_gen}_screp_{source_collection_sample_rep}_scssz_{source_collection_sample_size}.json",
        sample="sample_collection/source_sample_bsz_{burnin_size}_bmu_{burnin_mu}_bgen_{burnin_gen}_g_{genome_size}_brep_{burnin_sample_rep}_bssz_{burnin_sample_size}_ssz_{source_size}_smu_{source_mu}_sgen_{source_gen}_srep_{source_sample_rep}_sssz_{source_sample_size}_tsz_{transmission_size}_tmu_{transmission_mu}_tgen_{transmission_gen}_screp_{source_collection_sample_rep}_scssz_{source_collection_sample_size}.json"
    output: "sample_collection/source_sample_snp_matrix_bsz_{burnin_size}_bmu_{burnin_mu}_bgen_{burnin_gen}_g_{genome_size}_brep_{burnin_sample_rep}_bssz_{burnin_sample_size}_ssz_{source_size}_smu_{source_mu}_sgen_{source_gen}_srep_{source_sample_rep}_sssz_{source_sample_size}_tsz_{transmission_size}_tmu_{transmission_mu}_tgen_{transmission_gen}_screp_{source_collection_sample_rep}_scssz_{source_collection_sample_size}.csv"
    run:
        write_snp_matrix(input.sample, input.lineage, output[0])


