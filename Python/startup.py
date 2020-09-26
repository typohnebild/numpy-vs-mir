import optparse

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

    options, _ = parser.parse_args()
    return options
