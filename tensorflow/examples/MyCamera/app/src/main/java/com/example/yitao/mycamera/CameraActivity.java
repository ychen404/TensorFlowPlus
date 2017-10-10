package com.example.yitao.mycamera;

import android.Manifest;
import android.app.Activity;
import android.app.Fragment;
import android.media.ImageReader;
import android.os.Handler;
import android.os.HandlerThread;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.view.WindowManager;

import java.util.logging.Logger;

public class CameraActivity extends Activity implements ImageReader.OnImageAvailableListener {



    private static final Logger LOGGER = new Logger();

    private static final int PERMISSIONS_REQUEST = 1;

    private static final String PERMISSION_CAMERA = Manifest.permission.CAMERA;
    private static final String PERMISSION_STORAGE = Manifest.permission.WRITE_EXTERNAL_STORAGE;

    private boolean debug = false;

    private Handler handler;
    private HandlerThread handlerThread;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        LOGGER.d("onCreate " + this);
        super.onCreate(savedInstanceState);

        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
        setContentView(R.layout.activity_camera);

        if (hasPermission()) {
            setFragment();
        } else {
            requestPermission();
        }
    }

    protected void setFragment() {
        final Fragment fragment =
                CameraConnectionFragment.newInstance(

                )
    }

    private boolean hasPermission() {
    }


    @Override
    public void onImageAvailable(ImageReader imageReader) {

    }
}
