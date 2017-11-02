/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/


package org.tensorflow.demo;

import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.os.Environment;
import android.os.Trace;
import android.util.Log;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.List;
import java.util.PriorityQueue;
import java.util.StringTokenizer;
import java.util.Vector;
import org.tensorflow.Operation;
import org.tensorflow.contrib.android.TensorFlowInferenceInterface;

/** A classifier specialized to label images using TensorFlow. */
public class TensorFlowImageClassifier implements Classifier {
  private static final String TAG = "TensorFlowImageClassifier";

  // Only return this many results with at least this confidence.
  private static final int MAX_RESULTS = 3;
  private static final float THRESHOLD = 0.1f;

  // Config values.
  private String inputName;
  private String outputName;
  private int inputSize;
  private int imageMean;
  private float imageStd;
  private int batch = 128;
  private int total_pic = 1024;
  //private int bat = 4; // temp variable for grouping as a batch

  // Pre-allocated buffers.
  private Vector<String> labels = new Vector<String>();
  private int[] intValues;
  private float[] floatValues;
  private float[] outputs;
  private String[] outputNames;

  // For storing the size of an image in the dataset
  int width = 32;
  int height = 32;
  float[] inputSignals = new float[batch * inputSize];
  // float array for one image
  float smCon[];
  // float array for a batch
  float bigCon[];
  float[] labels_float = new float[batch];


  private boolean logStats = false;

  private TensorFlowInferenceInterface inferenceInterface;

  private TensorFlowImageClassifier() {}

  /**
   * Initializes a native TensorFlow session for classifying images.
   *
   * @param assetManager The asset manager to be used to load assets.
   * @param modelFilename The filepath of the model GraphDef protocol buffer.
   * @param labelFilename The filepath of label file for classes.
   * @param inputSize The input size. A square image of inputSize x inputSize is assumed.
   * @param imageMean The assumed mean of the image values.
   * @param imageStd The assumed std of the image values.
   * @param inputName The label of the image input node.
   * @param outputName The label of the output node.
   * @throws IOException
   */
  public static Classifier create(

          AssetManager assetManager,
          String modelFilename,
          String labelFilename,
          int inputSize,
          int imageMean,
          float imageStd,
          String inputName,
          String outputName) {
    TensorFlowImageClassifier c = new TensorFlowImageClassifier();
    c.inputName = inputName;
    c.outputName = outputName;

    // Read the label names into memory.
    // TODO(andrewharp): make this handle non-assets.
    String actualFilename = labelFilename.split("file:///android_asset/")[1];
    Log.i(TAG, "Reading labels from: " + actualFilename);
    BufferedReader br = null;
    try {
      br = new BufferedReader(new InputStreamReader(assetManager.open(actualFilename)));
      String line;
      while ((line = br.readLine()) != null) {
        c.labels.add(line);
      }
      br.close();
    } catch (IOException e) {
      throw new RuntimeException("Problem reading label file!" , e);
    }
    Log.d(TAG, "Classifier::create");
    c.inferenceInterface = new TensorFlowInferenceInterface(assetManager, modelFilename);

    // The shape of the output is [N, NUM_CLASSES], where N is the batch size.
    final Operation operation = c.inferenceInterface.graphOperation(outputName);
    Log.d(TAG, String.valueOf(operation.output(0)));

    final int numClasses = (int) operation.output(0).shape().size(0);
    // final int numClasses = 25;
    Log.i(TAG, "Read " + c.labels.size() + " labels, output layer size is " + numClasses);

    // Ideally, inputSize could have been retrieved from the shape of the input operation.  Alas,
    // the placeholder node for input in the graphdef typically used does not specify a shape, so it
    // must be passed in as a parameter.
    Log.d(TAG, "Classifier::inputSize");
    c.inputSize = inputSize;
    c.imageMean = imageMean;
    c.imageStd = imageStd;

    final int batch = 128;

    // Pre-allocate buffers.
    Log.d(TAG, "Classifier::Pre-allocate buffers");
    c.outputNames = new String[] {outputName};
    c.intValues = new int[32 * 32 * 3];
    //c.floatValues = new float[inputSize * inputSize * 3];
    c.floatValues = new float[inputSize * batch];
//    c.outputs = new float[numClasses];

//    c.outputs = new float[batch * outputsize];
    c.outputs = new float[batch];


    return c;
  }

  private float[] readImages() {
//    int intArray[];
//    float bmpFloatArray[];
//    float floatArray[];
    int name;
    int len = 0;
    smCon = new float[width * height * 3];
    bigCon = new float[width * height * 3 * batch];

    String fpath1;
    String filename;
    String fpath = Environment.getExternalStorageDirectory() + "/cifar/10000.png";
   // Log.d("readImages", String.valueOf(fpath));
    /*
    For reading all the files in a directory: start
     */
//    String path = Environment.getExternalStorageDirectory() + "/cifar/";
//    File dir = new File(path);
//    File[] directoryListing = dir.listFiles();

//    if (directoryListing != null) {
//      for (File child : directoryListing){
//          Log.d("readImages", String.valueOf(child.getName()));
//      }
//    } else {
//      // Handle problems
//    }

    /*
    For reading all the files in a directory: end
     */

    //for (int iter = 0; iter < total_pic/batch; iter++)
    for (int iter = 0; iter < 1; iter++)
    {
      for (int j = 0; j < batch; j++)
      {
     //   Log.d("readImages", "Inside for loop");
        name = 1 + iter * batch + j;

        fpath1 = Environment.getExternalStorageDirectory() + "/cifar/" + name + ".png";
     //   Log.d("readImages", "The file name is " + String.valueOf(fpath1));
        smCon = bmpToArray(fpath1);
     //   Log.d("readImages", "The smCon contains useful data");
        for (int k = 0; k < smCon.length; k ++, len++)
        {
          bigCon[len] = smCon[k];
        }
        readLabels(iter,j,batch);
      }
      len = 0;
      //tf related code

     // Log.d("readImages", "init label array");

//      float[] outSignals = new float[batch];
//      for ( int index = 0; index < batch; index ++) {
//        outSignals[index] = (float) Math.random();
//      }


      //writeToSDFile("Start batch size of " + String.valueOf(batch));
      inferenceInterface.runTarget(new String[] {"init_all_vars_op"});
        inferenceInterface.feed("input", bigCon, batch, inputSize);
    //    Log.d("readImages", "feed input");
        inferenceInterface.feed("label", labels_float, batch);
//      inferenceInterface.feed("label", outSignals, batch);

        inferenceInterface.run(new String[]{"loss"}, logStats);

        float[] resu = new float[1];
//      inferenceInterface.fetch("loss", resu);
        inferenceInterface.fetch("loss", resu);
        Log.d(TAG, "The loss is " + resu[0]);

        inferenceInterface.feed("input", bigCon, batch, inputSize);
        inferenceInterface.feed("label", labels_float, batch);
//      inferenceInterface.feed("label", outSignals, batch);
        //writeToSDFile("Start a new iteration");
        reportTime("training start:\t" + "iteration\t" + iter);
        inferenceInterface.runTarget(new String[]{"train"});
        reportTime("training ends:\t" + "iteration\t" + iter);
    //    writeToSDFile("End an iteration");
      //Log.d("readImages", "done training");

    }

    return bigCon;
  }


  private float[] bmpToArray(String fpath) {
    int intArray[];
    float bmpFloatArray[];
    float floatArray[];
    int name;

    Bitmap bmp = BitmapFactory.decodeFile(fpath);
  //  Log.d("readImages", "Read image");
    bmp = bmp.copy(Bitmap.Config.ARGB_8888, true);
//    width = bmp.getWidth();
//    height = bmp.getHeight();

    // Init array
    intArray = new int[bmp.getWidth() * bmp.getHeight()];
    bmpFloatArray = new float[bmp.getWidth() * bmp.getHeight()];
    bmp.getPixels(intArray, 0, bmp.getWidth(), 0, 0, bmp.getWidth(), bmp.getHeight());
  //  Log.d("readImages", "Init array");

    floatArray = new float[intArray.length * 3];
    for (int i = 0; i < intArray.length; i++)
    {
      // decode ARGB
      int A = (intArray[i] >> 24) & 0xff;
      int R = (intArray[i] >> 16) & 0xff;
      int G = (intArray[i] >>  8) & 0xff;
      int B = (intArray[i]      ) & 0xff;
      // floatArray[4*i] = (float) A;
      floatArray[3*i +0] = (float) R;
      floatArray[3*i +1] = (float) G;
      floatArray[3*i +2] = (float) B;
    }

    return floatArray;
  }


  private float[] readLabels(int idx_i, int idx_j, int batch) {
    // i and j are corresponding to the i and j from the outer loop
    String fpath = Environment.getExternalStorageDirectory() + "/label_number.csv";
    String csvFile = fpath;
    BufferedReader br = null;
    String line = "";
    String cvsSplitBy = ",";
    String[] labels_str;
   // float[] labels_float = new float[batch];

    try {
      br = new BufferedReader(new FileReader(csvFile));
      while ((line = br.readLine()) != null) {
        // use comma as separator
        labels_str = line.split(cvsSplitBy);
     //   System.out.println(labels_str.length);
     //   System.out.println("labels_str [code= " + labels_str[3] + " , name=" + labels_str[4] + "]");
     //   System.out.println(labels_str.length);
     //   labels_float = new float[labels_str.length];

          labels_float[idx_j] = Float.parseFloat(labels_str[idx_i * batch + idx_j]);
      }

    } catch (FileNotFoundException e) {
      e.printStackTrace();
    } catch (IOException e) {
      e.printStackTrace();
    } finally {
      if (br != null) {
        try {
          br.close();
        } catch (IOException e) {
          e.printStackTrace();
        }
      }
    }

  //  System.out.println(Arrays.toString(labels_float));
    return labels_float;
  }
  /*
  The method checkExternalMedia and writeToDisk are used to store the logcat
  info in a file so that the data processing will be easier.
     */
  /** Method to check whether external media available and writable. This is adapted from
   http://developer.android.com/guide/topics/data/data-storage.html#filesExternal */

  private void checkExternalMedia(){
    boolean mExternalStorageAvailable = false;
    boolean mExternalStorageWriteable = false;
    String state = Environment.getExternalStorageState();

    if (Environment.MEDIA_MOUNTED.equals(state)) {
      // Can read and write the media
      mExternalStorageAvailable = mExternalStorageWriteable = true;
    } else if (Environment.MEDIA_MOUNTED_READ_ONLY.equals(state)) {
      // Can only read the media
      mExternalStorageAvailable = true;
      mExternalStorageWriteable = false;
    } else {
      // Can't read or write
      mExternalStorageAvailable = mExternalStorageWriteable = false;
    }
    Log.d("writeTofile", "\n\nExternal Media: readable="   +mExternalStorageAvailable+" writable="+mExternalStorageWriteable);
    //tv.append("\n\nExternal Media: readable="   +mExternalStorageAvailable+" writable="+mExternalStorageWriteable);
  }

  private void writeToSDFile(String str){
    // Find the root of the external storage.
    // See http://developer.android.com/guide/topics/data/data-  storage.html#filesExternal

    File root = android.os.Environment.getExternalStorageDirectory();
    //tv.append("\nExternal file system root: "+root);
    Log.d("writeTofile", "\nExternal file system root: "+root);


    // See http://stackoverflow.com/questions/3551821/android-write-to-sd-card-folder
    File sdCard = Environment.getExternalStorageDirectory();
    File dir = new File (sdCard.getAbsolutePath() + "/measurements");
    //File dir = new File (root.getAbsolutePath() + "/sdcard");
    dir.mkdirs();
    File file = new File(dir, "myData.txt");

    try {
      FileOutputStream f = new FileOutputStream((file), true);
      PrintWriter pw = new PrintWriter(f);
      //pw.println("Hi , How are you");
      //pw.println("Hello");
      pw.println(str);
      pw.flush();
      pw.close();
      f.close();
    } catch (FileNotFoundException e) {
      e.printStackTrace();
      Log.i(TAG, "******* File not found. Did you" +
              " add a WRITE_EXTERNAL_STORAGE permission to the   manifest?");
    } catch (IOException e) {
      e.printStackTrace();
    }
    //tv.append("\n\nFile written to "+file);
    Log.d("writeTofile","\n\nFile written to "+file);

  }

  // Add a function to measure time
  long lastTime=0;
  public void reportTime(String str){
    long time = System.currentTimeMillis();
    long elapsed = time-lastTime;
    int Pid = android.os.Process.myPid();
    int Tid = android.os.Process.myTid();
    // Log.d("TFClassifier","Time elapsed:\t"+elapsed+"\t"+str+"\t"+"Current time: "+time + " Pid: " + Pid + " Tid: " + Tid );
    Log.d("TFClassifier","Time elapsed:\t"+elapsed+"\t\t"+str+"\t");
    writeToSDFile("Time elapsed:\t"+elapsed+"\t\t"+str+"\t");
    lastTime=System.currentTimeMillis();

  }

  @Override
  public List<Recognition> recognizeImage(final Bitmap bitmap) {
    // Log.d(TAG, "Classifier::recognizeImage");

    // Log this method so that it can be analyzed with systrace.
    Trace.beginSection("recognizeImage");

    Trace.beginSection("preprocessBitmap");

    // Preprocess the image data from 0-255 int to normalized float based
    // on the provided parameters.
    //Log.d(TAG, "Classifier::recognizeImage:preprocess");
//    bitmap.getPixels(intValues, 0, bitmap.getWidth(), 0, 0, bitmap.getWidth(), bitmap.getHeight());
//    for (int i = 0; i < intValues.length; ++i) {
//      final int val = intValues[i];
//      floatValues[i * 3 + 0] = (((val >> 16) & 0xFF) - imageMean) / imageStd;
//      floatValues[i * 3 + 1] = (((val >> 8) & 0xFF) - imageMean) / imageStd;
//      floatValues[i * 3 + 2] = ((val & 0xFF) - imageMean) / imageStd;
//    }
    Trace.endSection();

    checkExternalMedia();
    //reportTime("Create empty array");

    // Get the input from images
    float[] inputSignals = readImages();
//    readImages();

//    float[][][][] inputSignals = new float[batch][image_width][image_length][num_channels];
    float[] outSignals = new float[batch];
   // reportTime("Start filling array");
    // Assign image array to input
//    inputSignals = readImages();

    for ( int i = 0; i < inputSize * batch; i ++) {
      inputSignals[i] = (float) Math.random();
    }

    //Log.d(TAG, "Classifier::filled array");

    for ( int i = 0; i < batch; i ++) {
      outSignals[i] = (float) Math.random();
    }
    //reportTime("End filling array");
    inferenceInterface.runTarget(new String[] {"init_all_vars_op"});
    //reportTime("init_all_vars_op");
    //Log.d(TAG, "Classifier::variables initialization");


    for (int i = 0; i < 1; i++) {
      //Log.d(TAG, "The " + i + " iteration:");

      // Copy the input data into TensorFlow.
      //Log.d(TAG, "Classifier::feed");
//    Trace.beginSection("feed");
//    inferenceInterface.feed(inputName, floatValues, 1, inputSize, inputSize, 3);

    /*
    Need to make sure the content of src float array can feed the tensor with a size of the tensor
    For example, here float[] inputSignals is a [100 * 32] vector; the content is fed into a tensor
    of 100 (batch size defined here) * 32
     */


//      Feed x and y again to find the loss
//    inferenceInterface.feed("x", inputSignals, batch, 32);
//    inferenceInterface.feed("input", inputSignals, batch, inputSize);
      inferenceInterface.feed("input", inputSignals, batch, inputSize);
//      reportTime("feed input\t" + "iteration\t" + i);
     // Log.d(TAG, "Classifier::feed input");

//    inferenceInterface.feed("x", floatValues, batch, 32);
      // floatValues = new float[inputSize * 100];
      //  Log.d(TAG,"feed x");
      inferenceInterface.feed("label", outSignals, batch);
      reportTime("feed label\t" + "iteration\t" + i);
    //    Log.d(TAG, "Classifier::feed label");
//    inferenceInterface.feed("y", outputs, batch, 8);
      // outputs = new float[800];
      //  Log.d(TAG,"feed y");

//    Trace.endSection();

      // Run the inference call.
      //Log.d(TAG, "Classifier::recognizeImage:inferenceCall");

      //  Trace.beginSection("run");

//    inferenceInterface.run(outputNames, logStats);
      inferenceInterface.run(new String[]{"loss"}, logStats);

    //  reportTime("run loss\t" + "iteration\t" + i);
//      inferenceInterface.run(new String[]{"loss"}, logStats);
//      Trace.endSection();

      // Copy the output Tensor back into the output array.
      Log.d(TAG, "Classifier::fetch");
      Trace.beginSection("fetch");
//    inferenceInterface.fetch(outputName, outputs);
      float[] resu = new float[1];
//      inferenceInterface.fetch("loss", resu);
      inferenceInterface.fetch("loss", resu);
//      Log.d(TAG, "The loss of " + i + " iteration is " + resu[0]);

      Trace.endSection();
/*
  Feed x and y again for the training
 */

      inferenceInterface.feed("input", inputSignals, batch, inputSize);

      // Log.d(TAG, "feed x");
      inferenceInterface.feed("label", outSignals, batch);

      //  Log.d(TAG, "feed y");

    //  reportTime("training start dummy:\t" + "iteration\t" + i);
      inferenceInterface.runTarget(new String[]{"train"});
    //  reportTime("training end dummy:\t" + "iteration\t" + i);

    }
   /*Write to a file for plotting */

  //  writeToSDFile("Hello!!");

    // Find the best classifications.
    PriorityQueue<Recognition> pq =
            new PriorityQueue<Recognition>(
                    3,
                    new Comparator<Recognition>() {
                      @Override
                      public int compare(Recognition lhs, Recognition rhs) {
                        // Intentionally reversed to put high confidence at the head of the queue.
                        return Float.compare(rhs.getConfidence(), lhs.getConfidence());
                      }
                    });
    for (int i = 0; i < outputs.length; ++i) {
      if (outputs[i] > THRESHOLD) {
        pq.add(
                new Recognition(
                        "" + i, labels.size() > i ? labels.get(i) : "unknown", outputs[i], null));
      }
    }
    final ArrayList<Recognition> recognitions = new ArrayList<Recognition>();
    int recognitionsSize = Math.min(pq.size(), MAX_RESULTS);
    for (int i = 0; i < recognitionsSize; ++i) {
      recognitions.add(pq.poll());
    }
    Trace.endSection(); // "recognizeImage"
    return recognitions;
  }

  @Override
  public void enableStatLogging(boolean logStats) {
    this.logStats = logStats;
  }

  @Override
  public String getStatString() {
    return inferenceInterface.getStatString();
  }

  @Override
  public void close() {
    inferenceInterface.close();
  }
}