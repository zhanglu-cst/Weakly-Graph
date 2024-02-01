import matplotlib.pyplot as plt

x = list(range(1, 11))
baseline = [0.735, 0.71, 0.73, 0.80, 0.785, 0.885, 0.88, 0.87, 0.86, 0.87]
wo_ssl_only_dm = [0.743, 0.89, 0.845, 0.88, 0.875, 0.905, 0.885, 0.87, 0.87, 0.915]
wo_dm_only_ssl = [0.80, 0.905, 0.94, 0.93, 0.935, 0.925, 0.935, 0.925, 0.945, 0.925]
ours = [0.81, 0.915, 0.95, 0.94, 0.955, 0.945, 0.955, 0.945, 0.95, 0.955]

plt.figure(dpi = 600, figsize = (7, 6))
plt.subplot(3, 1, 1)

plt.plot(x, baseline, marker = 'v', linestyle = ':')
plt.plot(x, wo_ssl_only_dm, marker = '^', linestyle = '--')
plt.plot(x, wo_dm_only_ssl, marker = '+', linestyle = '-.')
plt.plot(x, ours, marker = 'o', linestyle = '-')
plt.xticks(list(range(1, 11)))
# plt.xlim((0.5, 10))
# plt.ylim((0, 11))

plt.ylabel('a) Classifier Accuracy')

plt.subplot(3, 1, 2)
baseline = [1, 0.83, 0.843, 0.844, 0.856, 0.853, 0.862, 0.85, 0.85, 0.843]
wo_ssl_only_dm = [1, 0.9, 0.91440501, 0.919753086, 0.91, 0.955684008, 0.956356736, 0.956603774, 0.956685499,
                  0.958724203, ]
wo_dm_only_ssl = [1, 0.95, 0.94, 0.93, 0.958490566, 0.930260223, 0.913369963, 0.933369963, 0.925201465, 0.941678832,
                  ]
ours = [1, 0.968680089, 0.965517241, 0.910358566, 0.958128655, 0.966090226, 0.977467167, 0.950520446, 0.967842779,
        0.968069217, ]
plt.plot(x, baseline, marker = 'v', linestyle = ':')
plt.plot(x, wo_ssl_only_dm, marker = '^', linestyle = '--')
plt.plot(x, wo_dm_only_ssl, marker = '+', linestyle = '-.')
plt.plot(x, ours, marker = 'o', linestyle = '-')
plt.xticks(list(range(1, 11)))
plt.ylabel('b) Pseudo Accuracy')

plt.subplot(3, 1, 3)
baseline = [0.081666667, 0.816666667, 0.826666667, 0.831666667, 0.836666667, 0.85, 0.875, 0.881666667, 0.888333333,
            0.893333333]
wo_ssl_only_dm = [0.081666667, 0.716666667, 0.798333333, 0.81, 0.833333333, 0.865, 0.878333333, 0.883333333, 0.885,
                  0.888333333]
wo_dm_only_ssl = [0.081666667, 0.728333333, 0.84, 0.858333333, 0.883333333, 0.896666667, 0.91, 0.91, 0.91,
                  0.913333333, ]
ours = [0.081666667, 0.745, 0.773333333, 0.836666667, 0.855, 0.886666667, 0.888333333, 0.896666667, 0.911666667,
        0.915, ]
p1 = plt.plot(x, baseline, marker = 'v', linestyle = ':', )
p2 = plt.plot(x, wo_ssl_only_dm, marker = '^', linestyle = '--')
p3 = plt.plot(x, wo_dm_only_ssl, marker = '+', linestyle = '-.')
p4 = plt.plot(x, ours, marker = 'o', linestyle = '-')
plt.xlabel('Iterations')
plt.ylabel('c) Pseudo Coverage')
plt.xticks(list(range(1, 11)))
plt.legend([p1, p2, p3, p4], labels = ['w/o SSL, w/o DM', 'w/o SSL, with DM', 'with SSL, w/o DM', 'ours'], loc = 'best')
# plt.show()

plt.savefig('ab_components.eps', dpi = 600, format = 'eps')
