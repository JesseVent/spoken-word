# python label_wav.py --graph=./graph/graph.pb --labels=../test/audiofiles.txt
#
# --wav=/tmp/speech_dataset/left/a5d485dc_nohash_0.wav
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import os

import pandas as pd
import tensorflow as tf

# pylint: disable=unused-import
from tensorflow.contrib.framework.python.ops import audio_ops as contrib_audio
# pylint: enable=unused-import

FLAGS = None

def load_graph(filename):
  """Unpersists graph from file as default graph."""
  with tf.gfile.FastGFile(filename, 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    tf.import_graph_def(graph_def, name='')


def load_labels(filename):
  """Read in labels, one label per line."""
  return [line.rstrip() for line in tf.gfile.GFile(filename)]

def load_wav(filename):
  """Read in labels, one label per line."""
  return [line.rstrip() for line in tf.gfile.GFile(filename)]


def run_graph(wav_data, labels, input_layer_name, output_layer_name,
              num_top_predictions):
  """Runs the audio data through the graph and prints predictions."""
  with tf.Session() as sess:
    # Feed the audio data as input to the graph.
    #   predictions  will contain a two-dimensional array, where one
    #   dimension represents the input image count, and the other has
    #   predictions per class
    softmax_tensor = sess.graph.get_tensor_by_name(output_layer_name)
    predictions, = sess.run(softmax_tensor, {input_layer_name: wav_data})

    # Sort to show labels in order of confidence
    top_k = predictions.argsort()[-num_top_predictions:][::-1]
    for node_id in top_k:
      human_string = labels[node_id]
      score = predictions[node_id]
      wav_result = ('%s (score = %.5f)' % (human_string, score))
      #wav_result = ('%s' % (human_string))
    return wav_result

def label_wav(wav, audio, labels, graph, input_name, output_name, how_many_labels):
    """Loads the model and labels, and runs the inference to print predictions."""

    labels_list = load_labels(labels)
    wav_list = load_wav(audio)
    load_graph(graph)
    fname = []
    wav_results = []
    for item in wav_list:
        audio_path = '../test/audio/'
        itemp = os.path.join(audio_path, item)
        with open(itemp, 'rb') as wav_file:
            wav_data = wav_file.read()
            wav_result = run_graph(wav_data, labels_list, input_name, output_name, how_many_labels)
        fname.append(item)
        print(wav_result)
        wav_results.append(wav_result)
    df = pd.DataFrame(columns=['fname', 'label'])
    df['fname'] = fname
    df['label'] = wav_results
    df.to_csv('spoken_submission_score.csv', index=False)

def main(_):
  """Entry point for script, converts flags to arguments."""
  label_wav(FLAGS.wav, FLAGS.audio, FLAGS.labels, FLAGS.graph, FLAGS.input_name,
            FLAGS.output_name, FLAGS.how_many_labels)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--wav', type=str, default='', help='Audio file to be identified.')
  parser.add_argument(
      '--graph', type=str, default='./graph/graph.pb', help='Model to use for identification.')
  parser.add_argument(
      '--audio', type=str, default='audio.txt', help='Path to file containing labels.')
  parser.add_argument(
      '--labels', type=str, default='conv_labels.txt', help='Path to file containing labels.')
  parser.add_argument(
      '--input_name',
      type=str,
      default='wav_data:0',
      help='Name of WAVE data input node in model.')
  parser.add_argument(
      '--output_name',
      type=str,
      default='labels_softmax:0',
      help='Name of node outputting a prediction in the model.')
  parser.add_argument(
      '--how_many_labels',
      type=int,
      default=1,
      help='Number of results to show.')

  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
