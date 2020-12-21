from os.path import join, abspath, dirname, pardir
BASE_DIR = abspath(join(dirname(__file__), pardir))
confdir   = join(BASE_DIR, 'conf.ini')
outputdir = join(BASE_DIR, 'dump')
LOG_FORMAT = "%(asctime)s %(name)-12s %(levelname)-8s %(message)s"

# Loss weight for gradient penalty
lambda_gp = 10

# for gan-glue
TRACE_SEP ='\t'