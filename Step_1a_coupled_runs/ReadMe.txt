WHAM_no_f4d:
#####################
We created this new coupled run as the first example in the transition to separating modes of operation into
"coupled" and "standalone" runs. We do not use the f4d file and all calculations are done thermally. We dont
calculate the sink term in FIDASIM as this is done in CQL3Dm

wham_source_test:
###################3
We have created this new directory to test the new version of the source text file. we are now providing both birth and sink points in this file and we want to test that we can read the file correcly in CQL3Dm.
We will use non-thermal f4d for calculting sinks
