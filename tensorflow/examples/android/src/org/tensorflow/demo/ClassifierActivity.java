
package org.tensorflow.demo;

import android.graphics.Bitmap;
import android.graphics.Bitmap.Config;
import android.graphics.Canvas;
import android.graphics.Matrix;
import android.graphics.Paint;
import android.graphics.Typeface;
import android.media.Image;
import android.media.Image.Plane;
import android.media.ImageReader;
import android.media.ImageReader.OnImageAvailableListener;
import android.os.SystemClock;
import android.os.Trace;
import android.util.Log;
import android.util.Size;
import android.util.TypedValue;
import android.view.Display;
import java.util.List;
import java.util.Vector;
import org.tensorflow.demo.OverlayView.DrawCallback;
import org.tensorflow.demo.env.BorderedText;
import org.tensorflow.demo.env.ImageUtils;
import org.tensorflow.demo.env.Logger;
import org.tensorflow.demo.R;

public class ClassifierActivity extends CameraActivity implements OnImageAvailableListener {
  private static final Logger LOGGER = new Logger();
  private static final String TAG = "ClassifierActivity";

  // These are the settings for the original v1 Inception model. If you want to
  // use a model that's been produced from the TensorFlow for Poets codelab,
  // you'll need to set IMAGE_SIZE = 299, IMAGE_MEAN = 128, IMAGE_STD = 128,
  // INPUT_NAME = "Mul", and OUTPUT_NAME = "final_result".
  // You'll also need to update the MODEL_FILE and LABEL_FILE paths to point to
  // the ones you produced.
  //
  // To use v3 Inception model, strip the DecodeJpeg Op from your retrained
  // model first:
  //
  // python strip_unused.py \
  // --input_graph=<retrained-pb-file> \
  // --output_graph=<your-stripped-pb-file> \
  // --input_node_names="Mul" \
  // --output_node_names="final_result" \
  // --input_binary=true
//  private static final int INPUT_SIZE = 224;
//  private static final int INPUT_SIZE = 32;
  private static final int INPUT_SIZE = 32 * 32 * 3;

  private static final int IMAGE_MEAN = 117;
  private static final float IMAGE_STD = 1;
  private static final String INPUT_NAME = "input";
  //  private static final String INPUT_NAME = "x";
//   private static final String OUTPUT_NAME = "y";
  private static final String OUTPUT_NAME = "label";


// private static final String MODEL_FILE = "file:///android_asset/mobilenet_bat_128_conv_5.pb";
// private static final String MODEL_FILE = "file:///android_asset/graph_conv_4_bat_128_w_256.pb";
// private static final String MODEL_FILE = "file:///android_asset/graph_conv_2_bat_128_w_256.pb";
// private static final String MODEL_FILE = "file:///android_asset/graph_fc_1_bat_32_w_64.pb";
 private static final String MODEL_FILE = "file:///android_asset/graph_full_bat_128_conv_width_512_fc_4096_noRelu_new.pb";
// private static final String MODEL_FILE = "file:///android_asset/graph_full_bat_32_conv_64_512_fc_1024_simplified.pb";
//private static final String MODEL_FILE = "file:///android_asset/graph_full_bat_32_conv_64_512_fc_2048_simplified.pb";
//  private static final String MODEL_FILE = "file:///android_asset/graph_saman.pb";

  private static final String LABEL_FILE =
          "file:///android_asset/imagenet_comp_graph_label_strings.txt";

  private static final boolean SAVE_PREVIEW_BITMAP = false;

  private static final boolean MAINTAIN_ASPECT = true;

  private static final Size DESIRED_PREVIEW_SIZE = new Size(640, 480);

  private Classifier classifier;

  private Integer sensorOrientation;

  private int previewWidth = 0;
  private int previewHeight = 0;
  private byte[][] yuvBytes;
  private int[] rgbBytes = null;
  private Bitmap rgbFrameBitmap = null;
  private Bitmap croppedBitmap = null;

  private Bitmap cropCopyBitmap;

  private boolean computing = false;

  private Matrix frameToCropTransform;
  private Matrix cropToFrameTransform;

  private ResultsView resultsView;

  private BorderedText borderedText;

  private long lastProcessingTimeMs;

  @Override
  protected int getLayoutId() {
    return R.layout.camera_connection_fragment;
  }

  @Override
  protected Size getDesiredPreviewFrameSize() {
    return DESIRED_PREVIEW_SIZE;
  }

  private static final float TEXT_SIZE_DIP = 10;

  @Override
  public void onPreviewSizeChosen(final Size size, final int rotation) {
    Log.d(TAG, "ClassifierActivity::onPreviewSizeChosen");
    final float textSizePx =
            TypedValue.applyDimension(
                    TypedValue.COMPLEX_UNIT_DIP, TEXT_SIZE_DIP, getResources().getDisplayMetrics());
    borderedText = new BorderedText(textSizePx);
    borderedText.setTypeface(Typeface.MONOSPACE);

    Log.d(TAG, "ClassifierActivity::classifier creation");

    classifier =
            TensorFlowImageClassifier.create(
                    getAssets(),
                    MODEL_FILE,
                    LABEL_FILE,
                    INPUT_SIZE,
                    IMAGE_MEAN,
                    IMAGE_STD,
                    INPUT_NAME,
                    OUTPUT_NAME);

    resultsView = (ResultsView) findViewById(R.id.results);
    previewWidth = size.getWidth();
    previewHeight = size.getHeight();


    Log.d(TAG, "ClassifierActivity::getDefaultDisplay");

    final Display display = getWindowManager().getDefaultDisplay();
    final int screenOrientation = display.getRotation();

    LOGGER.i("Sensor orientation: %d, Screen orientation: %d", rotation, screenOrientation);

    sensorOrientation = rotation + screenOrientation;

    LOGGER.i("Initializing at size %dx%d", previewWidth, previewHeight);
    rgbBytes = new int[previewWidth * previewHeight];
    rgbFrameBitmap = Bitmap.createBitmap(previewWidth, previewHeight, Config.ARGB_8888);
//    croppedBitmap = Bitmap.createBitmap(INPUT_SIZE, INPUT_SIZE, Config.ARGB_8888);

    Log.d(TAG, "ClassifierActivity::frameToCropTransform");

    frameToCropTransform =
            ImageUtils.getTransformationMatrix(
                    previewWidth, previewHeight,
                    INPUT_SIZE, INPUT_SIZE,
                    sensorOrientation, MAINTAIN_ASPECT);

    cropToFrameTransform = new Matrix();
    frameToCropTransform.invert(cropToFrameTransform);

    yuvBytes = new byte[3][];

    Log.d(TAG, "ClassifierActivity::addCallback");

    addCallback(
            new DrawCallback() {
              @Override
              public void drawCallback(final Canvas canvas) {
                renderDebug(canvas);
              }
            });
  }

  @Override
  public void onImageAvailable(final ImageReader reader) {
    //  Log.d(TAG, "onImageAvailable");
    Image image = null;

    try {
      image = reader.acquireLatestImage();

      if (image == null) {
        return;
      }

      if (computing) {
        image.close();
        return;
      }
      computing = true;

      Trace.beginSection("imageAvailable");

      final Plane[] planes = image.getPlanes();
      fillBytes(planes, yuvBytes);

      final int yRowStride = planes[0].getRowStride();
      final int uvRowStride = planes[1].getRowStride();
      final int uvPixelStride = planes[1].getPixelStride();
      ImageUtils.convertYUV420ToARGB8888(
              yuvBytes[0],
              yuvBytes[1],
              yuvBytes[2],
              previewWidth,
              previewHeight,
              yRowStride,
              uvRowStride,
              uvPixelStride,
              rgbBytes);

      image.close();
    } catch (final Exception e) {
      if (image != null) {
        image.close();
      }
      LOGGER.e(e, "Exception!");
      Trace.endSection();
      return;
    }

    rgbFrameBitmap.setPixels(rgbBytes, 0, previewWidth, 0, 0, previewWidth, previewHeight);
//    final Canvas canvas = new Canvas(croppedBitmap);
//    canvas.drawBitmap(rgbFrameBitmap, frameToCropTransform, null);

    // For examining the actual TF input.
    if (SAVE_PREVIEW_BITMAP) {
      ImageUtils.saveBitmap(croppedBitmap);
    }

    runInBackground(
            new Runnable() {
              @Override
              public void run() {
                final long startTime = SystemClock.uptimeMillis();
                final List<Classifier.Recognition> results = classifier.recognizeImage(croppedBitmap);
                lastProcessingTimeMs = SystemClock.uptimeMillis() - startTime;
		Log.i("TF_ANDROID_LOG", "Path time: " + lastProcessingTimeMs + "ms");

                cropCopyBitmap = Bitmap.createBitmap(croppedBitmap);
                resultsView.setResults(results);
                requestRender();
                computing = false;
              }
            });

    Trace.endSection();
  }

  @Override
  public void onSetDebug(boolean debug) {
    Log.d(TAG, "onSetDebug");
    classifier.enableStatLogging(debug);
  }

  private void renderDebug(final Canvas canvas) {
    Log.d(TAG, "renderDebug");
    if (!isDebug()) {
      return;
    }
    final Bitmap copy = cropCopyBitmap;
    if (copy != null) {
      final Matrix matrix = new Matrix();
      final float scaleFactor = 2;
      matrix.postScale(scaleFactor, scaleFactor);
      matrix.postTranslate(
              canvas.getWidth() - copy.getWidth() * scaleFactor,
              canvas.getHeight() - copy.getHeight() * scaleFactor);
      canvas.drawBitmap(copy, matrix, new Paint());

      final Vector<String> lines = new Vector<String>();
      if (classifier != null) {
        String statString = classifier.getStatString();
        String[] statLines = statString.split("\n");
        for (String line : statLines) {
          lines.add(line);
        }
      }

      lines.add("Frame: " + previewWidth + "x" + previewHeight);
      lines.add("Crop: " + copy.getWidth() + "x" + copy.getHeight());
      lines.add("View: " + canvas.getWidth() + "x" + canvas.getHeight());
      lines.add("Rotation: " + sensorOrientation);
      lines.add("Inference time: " + lastProcessingTimeMs + "ms");

      borderedText.drawLines(canvas, 10, canvas.getHeight() - 10, lines);
    }
  }
}
