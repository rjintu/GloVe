#!/bin/bash
set -e

# Makes programs, downloads sample data, trains a GloVe model, and then evaluates it.
# Requires a command line argument to specify training file
# One optional argument can specify the language used for eval script: matlab, octave or [default] python

make
# if [ ! -e text8 ]; then
#   if hash wget 2>/dev/null; then
#     wget http://mattmahoney.net/dc/text8.zip
#   else
#     curl -O http://mattmahoney.net/dc/text8.zip
#   fi
#   unzip text8.zip
#   rm text8.zip
# fi

CORPUS=$1
BASE=$(basename "$CORPUS")
VOCAB_FILE=vocab_${BASE%.*}.txt
COOCCURRENCE_FILE=cooccurrence_${BASE%.*}.bin
COOCCURRENCE_SHUF_FILE=cooccurrence_${BASE%.*}.shuf.bin
BUILDDIR=build
SAVE_FILE=vectors_${BASE%.*}
VERBOSE=2
MEMORY=4.0
VOCAB_MIN_COUNT=10
VECTOR_SIZE=200
MAX_ITER=15
WINDOW_SIZE=20
BINARY=2
NUM_THREADS=8
X_MAX=10
if hash python 2>/dev/null; then
    PYTHON=python
else
    PYTHON=python3
fi

echo
echo "$ $BUILDDIR/vocab_count -min-count $VOCAB_MIN_COUNT -verbose $VERBOSE < $CORPUS > $VOCAB_FILE"
$BUILDDIR/vocab_count -min-count $VOCAB_MIN_COUNT -verbose $VERBOSE < $CORPUS > $VOCAB_FILE
echo "$ $BUILDDIR/cooccur -memory $MEMORY -vocab-file $VOCAB_FILE -verbose $VERBOSE -window-size $WINDOW_SIZE < $CORPUS > $COOCCURRENCE_FILE"
$BUILDDIR/cooccur -memory $MEMORY -vocab-file $VOCAB_FILE -verbose $VERBOSE -window-size $WINDOW_SIZE < $CORPUS > $COOCCURRENCE_FILE
echo "$ $BUILDDIR/shuffle -memory $MEMORY -verbose $VERBOSE < $COOCCURRENCE_FILE > $COOCCURRENCE_SHUF_FILE"
$BUILDDIR/shuffle -memory $MEMORY -verbose $VERBOSE < $COOCCURRENCE_FILE > $COOCCURRENCE_SHUF_FILE
echo "$ rm $COOCCURRENCE_FILE" # remove the file to save space
rm $COOCCURRENCE_FILE
echo "$ $BUILDDIR/glove -save-file $SAVE_FILE -threads $NUM_THREADS -input-file $COOCCURRENCE_SHUF_FILE -x-max $X_MAX -iter $MAX_ITER -vector-size $VECTOR_SIZE -binary $BINARY -vocab-file $VOCAB_FILE -verbose $VERBOSE"
$BUILDDIR/glove -save-file $SAVE_FILE -threads $NUM_THREADS -input-file $COOCCURRENCE_SHUF_FILE -x-max $X_MAX -iter $MAX_ITER -vector-size $VECTOR_SIZE -binary $BINARY -vocab-file $VOCAB_FILE -verbose $VERBOSE
echo "$ rm $COOCCURRENCE_SHUF_FILE" # remove the file to save space
rm $COOCCURRENCE_SHUF_FILE

if [ "$CORPUS" = 'text8' ]; then
   if [ "$2" = 'matlab' ]; then
       matlab -nodisplay -nodesktop -nojvm -nosplash < ./eval/matlab/read_and_evaluate.m 1>&2 
   elif [ "$2" = 'octave' ]; then
       octave < ./eval/octave/read_and_evaluate_octave.m 1>&2
   else
       echo "$ $PYTHON eval/python/evaluate.py"
       $PYTHON eval/python/evaluate.py
   fi
fi
