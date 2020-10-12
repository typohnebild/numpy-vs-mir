import optparse
import time

DEFAULT_PROBLEM = '../problems/problem_2D_100.npy'


def getopts():
    parser = optparse.OptionParser()
    parser.add_option(
        '-n',
        action='store_true',
        dest='numba',
        default=False,
        help='activates numba')
    parser.add_option(
        '-v',
        action='store_true',
        dest='verbose',
        default=False,
        help='makes it more verbose')
    parser.add_option(
        '-d',
        action='store',
        dest='delay',
        type=int,
        default=500,
        help='delays the start of the run by DELAY ms (default:500)')

    parser.add_option('-p', action='store', dest='path',
                      default=DEFAULT_PROBLEM,
                      help='path to a problem (npy file) that is loaded')

    parser.add_option(
        '-t', action='store', dest='start_time',
        type='int',
        help='unix time stamp in nanoseconds of the programm call')

    options, _ = parser.parse_args()
    return options


def wait(options):
    rest = options.delay / 1000 - (time.time() - options.start_time / 1e9)

    if 0.1 < rest:
        time.sleep(rest)
    else:
        # if there is no time left over we can not be sure that the measurement
        # is not spoiled with startup so exit
        raise Exception("Warumup took to long")
