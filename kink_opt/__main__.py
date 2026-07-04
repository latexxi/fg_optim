import cProfile
import pstats
import sys

from .demos import main

if __name__ == "__main__":
    if "--profile" in sys.argv:
        profiler = cProfile.Profile()
        profiler.enable()
        try:
            main()
        finally:
            profiler.disable()
            stats = pstats.Stats(profiler).sort_stats("cumulative")
            stats.print_stats(30)
    else:
        main()
