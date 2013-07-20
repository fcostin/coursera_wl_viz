"""
data visualisation script for coursera warehouse location problem
"""

import numpy
import logging
from collections import namedtuple
import argparse


Problem = namedtuple('problem', ['n_hub', 'n_demand', 'hub_cap', 'hub_cost', 'demand', 'assign_cost'])

def parse(lines):
    t = lambda line : line.strip().split()

    header = t(lines[0])
    n_hubs, n_demands = int(header[0]), int(header[1])
    hub_cost = numpy.zeros((n_hubs, ), dtype=numpy.float)
    hub_cap = numpy.zeros((n_hubs, ), dtype=numpy.int)
    demand = numpy.zeros((n_demands, ), dtype=numpy.int)
    assign_cost = numpy.zeros((n_demands, n_hubs, ), dtype=numpy.float)

    for i in xrange(n_hubs):
        tokens = t(lines[i+1])
        assert len(tokens) == 2
        hub_cap[i] = int(tokens[0])
        hub_cost[i] = float(tokens[1])

    for i in xrange(n_demands):
        demand_tokens = t(lines[(2*i) + n_hubs + 1])
        assign_tokens = t(lines[(2*i + 1) + n_hubs + 1])
        assert len(demand_tokens) == 1 and len(assign_tokens) == n_hubs
        demand[i] = int(demand_tokens[0])
        assign_cost[i, :] = map(float, assign_tokens)

    return Problem(n_hubs, n_demands, hub_cap, hub_cost, demand, assign_cost)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('file_names', nargs='*')
    return p.parse_args()


def normalised(x):
    a, b = x.min(), x.max()
    if b == a:
        return 0 * x
    else:
        return (x - a) / (b - a)


def main():
    logging.basicConfig(level=logging.DEBUG)

    args = parse_args()

    import pylab
    for file_name in args.file_names:
        logging.info('reading "%s"' % file_name)
        with open(file_name, 'r') as f:
            problem = parse(list(f))
        logging.info('problem has %d hubs, %d demands' % (problem.n_hub, problem.n_demand))

        a = problem.assign_cost
        if False:
            for i in xrange(problem.n_demand):
                a[i, :] = normalised(a[i, :])

        a_mean = numpy.mean(a, axis=-1)

        a_mean_hub = numpy.mean(a, axis=0)

        tau = numpy.argsort(a_mean)
        a = a[tau]
        pylab.figure()
        pylab.suptitle(file_name)

        pylab.subplot(2, 2, 1)
        pylab.title('assignment costs (ordered by mean cost)')
        pylab.imshow(a, interpolation='nearest', cmap='Spectral_r')

        pylab.subplot(2, 2, 2)
        pylab.title('demands')
        pylab.scatter(problem.demand, a_mean)
        pylab.xlabel('demand')
        pylab.ylabel('mean assign cost')

        pylab.subplot(2, 2, 3)
        pylab.title('potential hubs')
        pylab.scatter(problem.hub_cost, problem.hub_cap)
        pylab.xlabel('build cost')
        pylab.ylabel('capacity')

        pylab.subplot(2, 2, 4)
        pylab.title('potential hubs')
        pylab.scatter(problem.hub_cost, a_mean_hub)
        pylab.xlabel('build cost')
        pylab.ylabel('mean assign cost')

    pylab.show()

if __name__ == '__main__':
    main()

