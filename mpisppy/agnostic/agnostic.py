# Agnostic.py
# This software is distributed under the 3-clause BSD License.

"""
notes by dlw:
   - The cfg will happen to have an Agnostic object added to it so mpisppy code can find it;
   however, this code does not care about that.
   - If a function in mpisppy has two callouts (a rarity), then the kwargs need to distinguish.
"""

import inspect
import pyomo.environ as pyo
from mpisppy.utils import sputils
from mpisppy.utils import config
import mpisppy.utils.solver_spec as solver_spec
import scenario_tree
import numpy as np


#========================================
class Agnostic():
    """
    Args:
        module (python module): None or the module with the callout fcts, 
                                scenario_creator and other helper fcts.
        cfg (Config): controls 
    """

    def __init__(self, module, cfg):
        self.module = module
        self.cfg = cfg

        
    def callout_agnostic(self, kwargs):
        """ callout from mpi-sppy for AML-agnostic support
        Args:
           cfg (Config): the field "AML_agnostic" might contain a module with callouts
	   kwargs (dict): the keyword args for the callout function (e.g. scenario)
        Calls:
           a callout function that presumably has side-effects
        Returns:
           True if the callout was done and False if not
        Note:
           Throws an error if the module exists, but the fct is missing
        """
       
        if self.module is not None:
            fname = inspect.stack()[1][3] 
            fct = getattr(self.module, fname, None)
            if fct is None:
                raise RuntimeError(f"AML-agnostic module {self.module.__name__} is missing function {fname}")
            fct(self, **kwargs)
            return True
        else:
            return False

       
    def scenario_creator(self, sname):
        """ create scenario sname by calling guest language, then attach stuff
        Args:
            sname (str): the scenario name that usually ends with a number
        Returns:
            scenario (Pyomo concrete model): a skeletal pyomo model with
                                             a lot attached to it.
        Note:
            The python function scenario_creator in the module needs to
            return a dict that we will call gd.
            gd["scenario"]: the guest language model handle
            gd["nonants"]: dict [(ndn,i)]: guest language Var handle
            gd["nonant_names"]: dict [(ndn,i)]: str with name of variable
            gd["nonant_fixedness"]: dict [(ndn,i)]: indicator of fixed variable
            gd["nonant_start"]: dict [(ndn,i)]: float with starting value
            gd["probability"]: float prob or str "uniform"
            gd["sense"]: pyo.minimize or pyo.maximize
            gd["BFs"]: scenario tree branching factors list or None
            (for two stage models, the only value of ndn is "ROOT";
             i in (ndn, i) is always just an index)
        """
        
        crfct = getattr(self.module, "scenario_creator", None)
        if crfct is None:
            raise RuntimeError(f"AML-agnostic module {self.module.__name__} is missing the scenario_creator function")
        kwfct = getattr(self.module, "kw_creator", None)
        if kwfct is not None:
           kwargs = kwfct(self.cfg)
           gd = crfct(sname, **kwargs)
        else:
            gd = crfct(sname)
        s = pyo.ConcreteModel(sname)

        ndns = [ndn for (ndn,i) in gd["nonants"].keys()]
        iis = [i for (ndn,i) in gd["nonants"].keys()]  # is is reserved...
        s.nonantVars = pyo.Var(ndns, iis)
        for idx,v  in s.nonantVars.items():
            v._value = gd["nonant_start"][idx]
            v.fixed = gd["nonant_fixedness"][idx]
        
        # we don't really need an objective, but we do need a sense
        # note that other code may put W's and prox's on it
        s.Obj = pyo.Objective(expr=0, sense=gd["sense"])
        s._agnostic_dict = gd

        if gd["BFs"] is None:
            assert gd["BFs"] is None, "We are only doing two stage for now"
            # (it would not be that hard to be multi-stage; see hydro.py)
            s._mpisppy_node_list = MakeNodesforScen(s, gd["BFs"], gd["nonants"])
            s._mpisppy_probability = 1/np.prod(gd["BFs"])
        else:
            sputils.attach_root_node(s, s.Obj, [s.nonantVars])

        s._mpisppy_probability = gd["probability"]
        
        return s

def MakeNodesforScen(model, BFs, var_dict):
    nonants_dict = {}
    for key in var_dict:
        ndn,i = key
        if ndn not in nonants_dict:
            nonants_dict[ndn] = list()
        nonants_dict[ndn].append(var_dict[key])
    node_list = [None] * len(nonants_dict)
    for ndn in nonants_dict:
        decomposed = ndn.split('_')
        l = len(decomposed)-1
        parent_name = '_'.join(decomposed[:-1])
        cost_expression = pyo.Expression(expr=0) # SHOULD BE MODIFIED
        if l == 0:
            node_list[0] = scenario_tree.ScenarioNode("ROOT",
                                         1.0,
                                         1,
                                         cost_expression,
                                         nonants_dict[ndn],
                                         model)
        else:
            node_list[l] = scenario_tree.ScenarioNode(
                name = ndn,
                cond_prob = BFs[l-1],
                stage = l+1,
                cost_expression = cost_expression,
                nonant_list = nonants_dict[ndn],
                scen_model=model,
                #nonant_ef_suppl_list = nonant_ef_suppl_list,
                parent_name = parent_name,
            )
    return node_list

############################################################################################################

        
if __name__ == "__main__":
    # For use by developers doing ad hoc testing
    print("no main")
