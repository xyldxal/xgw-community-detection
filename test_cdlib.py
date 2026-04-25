from cdlib import algorithms, ensemble, evaluation
print('Algorithms:')
print([x for x in dir(algorithms) if not x.startswith('_')])
print('Ensemble:')
print([x for x in dir(ensemble) if not x.startswith('_')])
