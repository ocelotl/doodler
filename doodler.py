from matplotlib.pyplot import plot, show, pause, close
from os.path import join
from ndjson import load as ndjson_load
from toml import load as toml_load

with open(
    join(
        toml_load('configuration.toml')['data']['path'],
        'quickdraw_simplified',
        'giraffe.ndjson'
    )
) as ndjson_file:

    data = ndjson_load(ndjson_file)

for datum in data:

    for x, y in datum['drawing']:

        plot(x, y, 'r-')

    show(block=False)
    pause(1)
    close()
