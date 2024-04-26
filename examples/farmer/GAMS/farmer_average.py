import os
import sys
import gams
import gamspy_base

this_dir = os.path.dirname(os.path.abspath(__file__))

gamspy_base_dir = gamspy_base.__path__[0]

w = gams.GamsWorkspace(working_directory=this_dir, system_directory=gamspy_base_dir)

model = w.add_job_from_file("farmer_average.gms")

model.run(output=sys.stdout)