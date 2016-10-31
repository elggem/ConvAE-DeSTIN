"""ConvAE Notepad (not for execution, just code snippets)"""

## In order to print the compiled compute graph:
theano.printing.pydotprint(train_0, outfile="./pics/symbolic_graph_opt_1.png", var_with_name_simple=True) 
theano.printing.pydotprint(train_1, outfile="./pics/symbolic_graph_opt_2.png", var_with_name_simple=True) 
theano.printing.pydotprint(train_2, outfile="./pics/symbolic_graph_opt_3.png", var_with_name_simple=True) 
theano.printing.pydotprint(train_3, outfile="./pics/symbolic_graph_opt_4.png", var_with_name_simple=True) 

