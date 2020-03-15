/*
 * Copyright 2019 The TensorFlow Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *       http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/* Demonstrates how to run an audio recognition model in Android.

This example loads a simple speech recognition model trained by the tutorial at
https://www.tensorflow.org/tutorials/audio_training

The model files should be downloaded automatically from the TensorFlow website,
but if you have a custom model you can update the LABEL_FILENAME and
MODEL_FILENAME constants to point to your own files.

The example application displays a list view with all of the known audio labels,
and highlights each one when it thinks it has detected one through the
microphone. The averaging of results to give a more reliable signal happens in
the RecognizeCommands helper class.
*/

package org.tensorflow.lite.examples.speech;

import android.app.Activity;
import android.content.Context;
import android.content.pm.PackageManager;
import android.content.res.AssetFileDescriptor;
import android.content.res.AssetManager;
import android.media.AudioFormat;
import android.media.AudioRecord;
import android.media.MediaRecorder;
import android.os.Build;
import android.os.Bundle;
import android.os.Handler;
import android.os.HandlerThread;
import androidx.annotation.NonNull;
import com.google.android.material.bottomsheet.BottomSheetBehavior;
import androidx.appcompat.widget.SwitchCompat;

import android.preference.TwoStatePreference;
import android.util.Log;
import android.view.View;
import android.view.ViewTreeObserver;
import android.widget.CompoundButton;
import android.widget.ImageView;
import android.widget.LinearLayout;
import android.widget.TextView;
import java.io.ByteArrayOutputStream;
import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.IOException;

import java.io.InputStreamReader;
import java.io.InputStream;
import java.io.File;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.io.FileOutputStream;
import java.io.OutputStream;
import java.io.RandomAccessFile;

import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Date;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.locks.ReentrantLock;
// 이 파일은 어디에 있는거여?!?!?
import org.tensorflow.lite.Interpreter;
// MFCC
import org.tensorflow.demo.mfcc.MFCC;

/**
 * An activity that listens for audio and then uses a TensorFlow model to detect particular classes,
 * by default a small set of action words.
 */
public class SpeechActivity extends Activity
    implements View.OnClickListener, CompoundButton.OnCheckedChangeListener {

  // Constants that control the behavior of the recognition code and model
  // settings. See the audio recognition tutorial for a detailed explanation of
  // all these, but you should customize them to match your training settings if
  // you are running your own model.

//  private static final int SAMPLE_RATE = 16000; !!
  private static final int SAMPLE_RATE = 22050;
//  private static final int SAMPLE_DURATION_MS = 1000; !!
  private static final int SAMPLE_DURATION_MS = 10000;
  private static final int RECORDING_LENGTH = (int) (SAMPLE_RATE * SAMPLE_DURATION_MS / 1000);
  // !! 여기 새로 생김
  private static final long AVERAGE_WINDOW_DURATION_MS = 1000;
  private static final float DETECTION_THRESHOLD = 0.50f;
  private static final int SUPPRESSION_MS = 1500;
  private static final int MINIMUM_COUNT = 3;
  private static final long MINIMUM_TIME_BETWEEN_SAMPLES_MS = 30;
  // !!
//  private static final String LABEL_FILENAME = "file:///android_asset/conv_actions_labels.txt";
  private static final String LABEL_FILENAME = "file:///android_asset/labels.txt";
//  private static final String MODEL_FILENAME = "file:///android_asset/conv_actions_frozen.tflite";
  private static final String MODEL_FILENAME = "file:///android_asset/converted_model.tflite";

  // UI elements.
  private static final int REQUEST_RECORD_AUDIO = 13;
  private static final String LOG_TAG = SpeechActivity.class.getSimpleName();

  // Working variables.
  short[] recordingBuffer = new short[RECORDING_LENGTH];
  byte[] recordingByteBuffer = new byte[RECORDING_LENGTH*2];
  int recordingOffset = 0;
  boolean shouldContinue = true;
  private Thread recordingThread;
  boolean shouldContinueRecognition = true;
  private Thread recognitionThread;
  // mutex lock 같은거네!!
  private final ReentrantLock recordingBufferLock = new ReentrantLock();
  //!! 여기 새로 생김
  private List<String> labels = new ArrayList<String>();
  private List<String> displayedLabels = new ArrayList<>();
  private RecognizeCommands recognizeCommands = null;
  private LinearLayout bottomSheetLayout;
  private LinearLayout gestureLayout;
  private BottomSheetBehavior<LinearLayout> sheetBehavior;
  /// !! 새로생김 끝

  ////!!!!!!!!!!!!!!!!!!!!!!!!! 이걸로 정의
  private Interpreter tfLite;
  private FileOutputStream outputStream;

  // !! 여기 새로 생
  private ImageView bottomSheetArrowImageView;

  private TextView yesTextView,
      noTextView,
      upTextView,
      downTextView,
      leftTextView,
      rightTextView,
      onTextView,
      offTextView,
      stopTextView,
      goTextView;
  private TextView sampleRateTextView, inferenceTimeTextView;
  private ImageView plusImageView, minusImageView;
  private SwitchCompat apiSwitchCompat;
  private TextView threadsTextView;
  private long lastProcessingTimeMs;
  private Handler handler = new Handler();
  private TextView selectedTextView = null;
  private HandlerThread backgroundThread;
  private Handler backgroundHandler;
  // !! 새로생김

  /** Memory-map the model file in Assets. */
  private static MappedByteBuffer loadModelFile(AssetManager assets, String modelFilename)
      throws IOException {
    AssetFileDescriptor fileDescriptor = assets.openFd(modelFilename);
    FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
    FileChannel fileChannel = inputStream.getChannel();
    long startOffset = fileDescriptor.getStartOffset();
    long declaredLength = fileDescriptor.getDeclaredLength();
    return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
  }

  @Override
  protected void onCreate(Bundle savedInstanceState) {
    // Set up the UI.
    super.onCreate(savedInstanceState);
    setContentView(R.layout.activity_speech);

    // Load the labels for the model, but only display those that don't start
    // with an underscore.
    String actualLabelFilename = LABEL_FILENAME.split("file:///android_asset/", -1)[1];
    Log.i(LOG_TAG, "Reading labels from: " + actualLabelFilename);
    BufferedReader br = null;
    try {
      br = new BufferedReader(new InputStreamReader(getAssets().open(actualLabelFilename)));
      String line;
      while ((line = br.readLine()) != null) {
        labels.add(line);
        if (line.charAt(0) != '_') {
          displayedLabels.add(line.substring(0, 1).toUpperCase() + line.substring(1));
        }
      }
      br.close();
    } catch (IOException e) {
      throw new RuntimeException("Problem reading label file!", e);
    }

    // Set up an object to smooth recognition results to increase accuracy.
    recognizeCommands =
        new RecognizeCommands(
            labels,
            AVERAGE_WINDOW_DURATION_MS,
            DETECTION_THRESHOLD,
            SUPPRESSION_MS,
            MINIMUM_COUNT,
            MINIMUM_TIME_BETWEEN_SAMPLES_MS);


    // !! 여기서 모델 파일 받아서 실험함!!
    String actualModelFilename = MODEL_FILENAME.split("file:///android_asset/", -1)[1];
    try {
      tfLite = new Interpreter(loadModelFile(getAssets(), actualModelFilename));
    } catch (Exception e) {
      throw new RuntimeException(e);
    }

    // RECORDING_LENGTH는 220500
    // array([  1,  50, 431,   3], dtype=int32) 가 input shape였음
    // array([1, 2], dtype=int32) output shape!
    //!! 여기부분 수정! (IDx는 모르겠다)
    tfLite.resizeInput(0, new int[] {1, 50, 431, 3});
//    tfLite.resizeInput(1, new int[] {1});

    // Start the recording and recognition threads.
    requestMicrophonePermission();
    startRecording();
//    startRecognition();

    sampleRateTextView = findViewById(R.id.sample_rate);
    inferenceTimeTextView = findViewById(R.id.inference_info);
    bottomSheetLayout = findViewById(R.id.bottom_sheet_layout);
    gestureLayout = findViewById(R.id.gesture_layout);
    sheetBehavior = BottomSheetBehavior.from(bottomSheetLayout);
    bottomSheetArrowImageView = findViewById(R.id.bottom_sheet_arrow);

    threadsTextView = findViewById(R.id.threads);
    plusImageView = findViewById(R.id.plus);
    minusImageView = findViewById(R.id.minus);
    apiSwitchCompat = findViewById(R.id.api_info_switch);

    yesTextView = findViewById(R.id.yes);
    noTextView = findViewById(R.id.no);
    upTextView = findViewById(R.id.up);
    downTextView = findViewById(R.id.down);
    leftTextView = findViewById(R.id.left);
    rightTextView = findViewById(R.id.right);
    onTextView = findViewById(R.id.on);
    offTextView = findViewById(R.id.off);
    stopTextView = findViewById(R.id.stop);
    goTextView = findViewById(R.id.go);

    apiSwitchCompat.setOnCheckedChangeListener(this);

    ViewTreeObserver vto = gestureLayout.getViewTreeObserver();
    vto.addOnGlobalLayoutListener(
        new ViewTreeObserver.OnGlobalLayoutListener() {
          @Override
          public void onGlobalLayout() {
            gestureLayout.getViewTreeObserver().removeOnGlobalLayoutListener(this);
            int height = gestureLayout.getMeasuredHeight();

            sheetBehavior.setPeekHeight(height);
          }
        });
    sheetBehavior.setHideable(false);

    sheetBehavior.setBottomSheetCallback(
        new BottomSheetBehavior.BottomSheetCallback() {
          @Override
          public void onStateChanged(@NonNull View bottomSheet, int newState) {
            switch (newState) {
              case BottomSheetBehavior.STATE_HIDDEN:
                break;
              case BottomSheetBehavior.STATE_EXPANDED:
                {
                  bottomSheetArrowImageView.setImageResource(R.drawable.icn_chevron_down);
                }
                break;
              case BottomSheetBehavior.STATE_COLLAPSED:
                {
                  bottomSheetArrowImageView.setImageResource(R.drawable.icn_chevron_up);
                }
                break;
              case BottomSheetBehavior.STATE_DRAGGING:
                break;
              case BottomSheetBehavior.STATE_SETTLING:
                bottomSheetArrowImageView.setImageResource(R.drawable.icn_chevron_up);
                break;
            }
          }

          @Override
          public void onSlide(@NonNull View bottomSheet, float slideOffset) {}
        });

    plusImageView.setOnClickListener(this);
    minusImageView.setOnClickListener(this);

    sampleRateTextView.setText(SAMPLE_RATE + " Hz");
  }

  private void requestMicrophonePermission() {
    if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
      requestPermissions(
          new String[] {android.Manifest.permission.RECORD_AUDIO}, REQUEST_RECORD_AUDIO);
    }
  }

  @Override
  public void onRequestPermissionsResult(
      int requestCode, String[] permissions, int[] grantResults) {
    if (requestCode == REQUEST_RECORD_AUDIO
        && grantResults.length > 0
        && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
//      startRecording();
//      startRecognition();
    }
  }

  public synchronized void startRecording() {
    if (recordingThread != null) {
      return;
    }
    shouldContinue = true;
    recordingThread =
        new Thread(
            new Runnable() {
              @Override
              public void run() {
                record();
              }
            });
    recordingThread.start();
  }

  public synchronized void stopRecording() {
    if (recordingThread == null) {
      return;
    }
    shouldContinue = false;
    recordingThread = null;
  }

  ////////////////////////


  public static void writeWavHeader(OutputStream out, short channels, int sampleRate, short bitDepth) throws IOException {
    // WAV 포맷에 필요한 little endian 포맷으로 다중 바이트의 수를 raw byte로 변환한다.
    byte[] littleBytes = ByteBuffer
            .allocate(14)
            .order(ByteOrder.LITTLE_ENDIAN)
            .putShort(channels)
            .putInt(sampleRate)
            .putInt(sampleRate * channels * (bitDepth / 8))
            .putShort((short) (channels * (bitDepth / 8)))
            .putShort(bitDepth)
            .array();
    // 최고를 생성하지는 않겠지만, 적어도 쉽게만 가자.
    out.write(new byte[]{
            'R', 'I', 'F', 'F', // Chunk ID
            0, 0, 0, 0, // Chunk Size (나중에 업데이트 될것)
            'W', 'A', 'V', 'E', // Format
            'f', 'm', 't', ' ', //Chunk ID
            16, 0, 0, 0, // Chunk Size
            1, 0, // AudioFormat
            littleBytes[0], littleBytes[1], // Num of Channels
            littleBytes[2], littleBytes[3], littleBytes[4], littleBytes[5], // SampleRate
            littleBytes[6], littleBytes[7], littleBytes[8], littleBytes[9], // Byte Rate
            littleBytes[10], littleBytes[11], // Block Align
            littleBytes[12], littleBytes[13], // Bits Per Sample
            'd', 'a', 't', 'a', // Chunk ID
            0, 0, 0, 0, //Chunk Size (나중에 업데이트 될 것)
    });
  }

  public static void updateWavHeader(File wav) throws IOException {
    byte[] sizes = ByteBuffer
            .allocate(8)
            .order(ByteOrder.LITTLE_ENDIAN)
            // 아마 이 두 개를 계산할 때 좀 더 좋은 방법이 있을거라 생각하지만..
            .putInt((int) (wav.length() - 8)) // ChunkSize
            .putInt((int) (wav.length() - 44)) // Chunk Size
            .array();
    RandomAccessFile accessWave = null;
    try {
      accessWave = new RandomAccessFile(wav, "rw"); // 읽기-쓰기 모드로 인스턴스 생성
      // ChunkSize
      accessWave.seek(4); // 4바이트 지점으로 가서
      accessWave.write(sizes, 0, 4); // 사이즈 채움
      // Chunk Size
      accessWave.seek(40); // 40바이트 지점으로 가서
      accessWave.write(sizes, 4, 4); // 채움
    } catch (IOException ex) {
      // 예외를 다시 던지나, finally 에서 닫을 수 있음
      throw ex;
    } finally {
      if (accessWave != null) {
        try {
          accessWave.close();
        } catch (IOException ex) {
          // 무시
        }
      }
    }
  }

  private void processCapture(byte[] buffer, int status) {
    if (status == AudioRecord.ERROR_INVALID_OPERATION || status == AudioRecord.ERROR_BAD_VALUE)
      return;
    try {
      outputStream.write(buffer, 0, buffer.length);
      Log.v(LOG_TAG, "===================> WRITE" + buffer.length);
    } catch (IOException e) {
      e.printStackTrace();
    }
  }

  ////////////////////////

  private void record() {
    android.os.Process.setThreadPriority(android.os.Process.THREAD_PRIORITY_AUDIO);

    // Estimate the buffer size we'll need for this device.
    // 이걸로 buffersize를 얻는다!!!!
    int bufferSize =  AudioRecord.getMinBufferSize(
                        SAMPLE_RATE, AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT);

    if (bufferSize == AudioRecord.ERROR || bufferSize == AudioRecord.ERROR_BAD_VALUE) {
      bufferSize = SAMPLE_RATE * 2;
    }

    // ?? SAMPLE RATE 만큼의 audio Buffer를 만드는건가여...?
    // byte로 들어오면 short로 바꿔지니까 뭉쳐져서 그런듯? (16bit니까)
    short[] audioBuffer = new short[bufferSize / 2];


    // !!! 이걸로 record initialize
    AudioRecord record =
        new AudioRecord(
            MediaRecorder.AudioSource.DEFAULT,
            SAMPLE_RATE,
            AudioFormat.CHANNEL_IN_MONO,
            AudioFormat.ENCODING_PCM_16BIT,
            bufferSize);

    if (record.getState() != AudioRecord.STATE_INITIALIZED) {
      Log.e(LOG_TAG, "Audio Record can't initialize!");
      return;
    }

    record.startRecording();

    Log.v(LOG_TAG, "Start recording");

    // Loop, gathering audio data and copying it to a round-robin buffer.
    while (shouldContinue) {
      // 얼마나 읽었
      int numberRead = record.read(audioBuffer, 0, audioBuffer.length);
      int maxLength = recordingBuffer.length;
      int newRecordingOffset = recordingOffset + numberRead;

//      int secondCopyLength = Math.max(0, newRecordingOffset - maxLength);
//      int firstCopyLength = numberRead - secondCopyLength;
      // We store off all the data for the recognition thread to access. The ML
      // thread will copy out of this buffer into its own, while holding the
      // lock, so this should be thread safe.
      // recordingbuffer와 audioBuffer차이 audiobuffer에 읽어온걸 recording buffer에 잘라서 넣
      // mutextLock 같은거!!
      recordingBufferLock.lock();
      try {
//        // arraycopy(Object src, int srcPos, Object dest, int destPos, int length)음
//        // 아 round robin으로 만드려고 앞뒤를 자르는거구나!!! 이해이해
//        System.arraycopy(audioBuffer, 0, recordingBuffer, recordingOffset, firstCopyLength); // 여기서 0
//        System.arraycopy(audioBuffer, firstCopyLength, recordingBuffer, 0, secondCopyLength);
//        recordingOffset = newRecordingOffset % maxLength;
          if (recordingOffset + numberRead < maxLength) {
            // audio buffer를 붙여넣
            Log.d(LOG_TAG, "audiobuffer: " + audioBuffer.length +" "+  Arrays.toString(audioBuffer));
            System.arraycopy(audioBuffer, 0, recordingBuffer, recordingOffset, numberRead);
          } else {
            try {
              // 내부저장소 외부저장소(https://codechacha.com/ko/android-q-scoped-storage/)
              // EXTERNAL 시도 -> Android/data/com.~~/ 폴더에 있
              File file = new File(getExternalFilesDir(null), "test.wav");
              outputStream = new FileOutputStream(file);

              Log.d(LOG_TAG, "External file dir: "
                      + getExternalFilesDir(null));

              // 여기는 한 버퍼만 들어가니까 10초 넘게 하려면 한참 더해야 되는건가??
              Log.d(LOG_TAG, "recordingbufferinfo: " +recordingBuffer[1800]); // 값이 적히긴 하는데/...
              Log.d(LOG_TAG, "recordingbufferinfo: " + recordingBuffer.length +" "+  Arrays.toString(recordingBuffer));
              writeWavHeader(outputStream,(short)AudioFormat.CHANNEL_IN_MONO, (short)SAMPLE_RATE, (short)AudioFormat.ENCODING_PCM_16BIT);
              // 여기에 파일 씀
              Log.d(LOG_TAG, "Number Byte Read: " + numberRead);

              // short to byte(recording buffer to
              // https://stackoverflow.com/questions/10804852/how-to-convert-short-array-to-byte-array
              ByteBuffer byteBuf = ByteBuffer.allocate(2* recordingBuffer.length);
              int i=0;
              while (recordingBuffer.length > i) {
                byteBuf.putShort(recordingBuffer[i]);
                i++;
              }

              // !! 결과가 0으로 나옴
             // https://medium.com/@ponychen/java-bytebuffer-to-byte-array-a347d5c3a576
//              byte[] bytes = new byte[byteBuf.remaining()];
//              byteBuf.get(bytes, 0, bytes.length);
              Log.d(LOG_TAG, "bytesinfo: "  +  Arrays.toString(byteBuf.array()));
              Log.d(LOG_TAG, "bytes length info: "  +  byteBuf.array().length);

              processCapture(byteBuf.array(), numberRead); // recordingBuffer로 해야함!!(10초 버퍼 채워진크기)
              // wav header 붙이
              updateWavHeader(file);


//        outputStream.close();
            } catch (IOException e) {
              e.printStackTrace();
            }
            shouldContinue = false;
          }
          recordingOffset += numberRead;
      } finally {
        recordingBufferLock.unlock();
      }
    }

    record.stop();
    record.release();
    startRecognition();
  }

  public synchronized void startRecognition() {
    if (recognitionThread != null) {
      return;
    }
    shouldContinueRecognition = true;
    recognitionThread =
        new Thread(
            new Runnable() {
              @Override
              public void run() {
                recognize();
              }
            });
    recognitionThread.start();
  }

  public synchronized void stopRecognition() {
    if (recognitionThread == null) {
      return;
    }
    shouldContinueRecognition = false;
    recognitionThread = null;
  }

  public byte[] inputStreamToByteArray(InputStream inStream) throws IOException {
    ByteArrayOutputStream baos = new ByteArrayOutputStream();
    byte[] buffer = new byte[8192];
    int bytesRead;
    while ((bytesRead = inStream.read(buffer)) > 0) {
      baos.write(buffer, 0, bytesRead);
    }
    return baos.toByteArray();
  }

  private void recognize() {

    Log.v(LOG_TAG, "Start recognition");


    short[] inputBuffer = new short[RECORDING_LENGTH];
    float[][] floatInputBuffer = new float[RECORDING_LENGTH][1];
    // !! Double로 받아야 하는듯
    double[] doubleInputBuffer = new double[RECORDING_LENGTH];
//    float[][] outputScores = new float[1][labels.size()];
    float[][] outputScores = new float[1][2];
    int[] sampleRateList = new int[] {SAMPLE_RATE};

    // Loop, grabbing recorded data and running the recognition model on it.
    // !!! 계속 돌아가는데 한번만 돌악게 해보자
//    while (shouldContinueRecognition) {
      long startTime = new Date().getTime();
//      // The recording thread places data in this round-robin buffer, so lock to
//      // make sure there's no writing happening and then copy it to our own
//      // local version.
//      recordingBufferLock.lock();
//      try {
//        int maxLength = recordingBuffer.length;
//        int firstCopyLength = maxLength - recordingOffset;
//        int secondCopyLength = recordingOffset;
//        System.arraycopy(recordingBuffer, recordingOffset, inputBuffer, 0, firstCopyLength);
//        System.arraycopy(recordingBuffer, 0, inputBuffer, firstCopyLength, secondCopyLength);
//      } finally {
//        recordingBufferLock.unlock();
//      }
    recordingBufferLock.lock();
    try {
      int maxLength = recordingBuffer.length;
      System.arraycopy(recordingBuffer, 0, inputBuffer, 0, maxLength);
    } finally {
      recordingBufferLock.unlock();
    }

    /////////////////////////////////////////


    /////////////////////////////////////////
//      // 여기에서 short로 되어있는 input Buffer 만들자.
//      // read asset to File
//
//    ByteArrayOutputStream buffer = new ByteArrayOutputStream();
//    try{
//      // file to InputStream
//      // https://evnt-hrzn.tistory.com/23
//        AssetManager assetManager = getAssets();
//        InputStream is = assetManager.open("8662.mp3");
//      // InputStream to Byte
//      // https://stackoverflow.com/questions/1264709/convert-inputstream-to-byte-array-in-java
//        int nRead;
//        byte[] data = new byte[220500];
//
//        while ((nRead = is.read(data, 0, data.length)) != -1) {
//          buffer.write(data, 0, nRead);
//        }
//
//
//      }catch(IOException e){
//        e.printStackTrace();
//      }
//    byte[] bytes =buffer.toByteArray();
//
//    // https://stackoverflow.com/questions/47790970/convert-byte-array-to-short-array
//    short[] newinputBuffer = new short[bytes.length/2];
//    ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN).asShortBuffer().get(newinputBuffer);
//
//    // 80421
//    Log.v(LOG_TAG, "NEW_INPUTBUFFER_SIZE======> " + newinputBuffer.length);
//    // 22050(10초 단위로 load)
//    Log.v(LOG_TAG, "RECORDINGLENGTH======> " + RECORDING_LENGTH);
//
//      // We need to feed in float values between -1.0f and 1.0f, so divide the
//      // signed 16-bit inputs.
//      for (int i = 0; i < RECORDING_LENGTH; ++i) {
//        doubleInputBuffer[i] = newinputBuffer[i] / 32767.0f;
//      }
//
      /////////////////////////////////////////
    // !!! SAMPE RATE에 맞춰서 load 해줘야 한다
    for (int i = 0; i < RECORDING_LENGTH; ++i) {
        doubleInputBuffer[i] = inputBuffer[i] / 32767.0f;
      }

      //MFCC java library.
      // MFCC 로바꾸기
      MFCC mfccConvert = new MFCC();
      float[] mfccInput = mfccConvert.process(doubleInputBuffer);
        //!! 이렇게 repeat 하는거 아니다!
//        // 3 channel 이라 3번 repeat 하기듯
//        // ?? 이거 좀 더 좋게 할 방법!!(이것도 잘 된건지 모르겠어 A[2:3] 이런건 없나...)
//        mfccInput = Arrays.copyOf(mfccInput, mfccInput.length * 3);
//        for(int i=mfccInput.length/3; i<mfccInput.length/3 * 2; i++){
//            Log.v(LOG_TAG, "MFCC Input======> " +  mfccInput[i-mfccInput.length/3]);
//            mfccInput[i] = mfccInput[i-mfccInput.length/3];
//        }
//        for(int i=mfccInput.length/3*2; i<mfccInput.length/3 * 3; i++){
//            mfccInput[i] = mfccInput[i-mfccInput.length*2/3];
//        }

      // 5초따리
      // 16000으로 하면 3140 이 원래 input 근데 22050으로 바꾸면 4320 둘다 0.19
      // 그니까 5초에 160000이면 157 22050이면 216
      // 10초면 432니까 비슷비슷??
      // 그 1차원임! 157X20 = 3140
      // mfcc class에서 직접 sampling rate 바꿔야 함 -> 그러면 상이즈 맞게 나옴!
      Log.v(LOG_TAG, "MFCC_SIZE======> " + mfccInput.length);
      Log.v(LOG_TAG, "MFCC Input======> " + Arrays.toString(mfccInput));
    Log.v(LOG_TAG, "MFCC Input======> " + mfccInput[0]);

      float sum = 0;
      for (int i=0; i<mfccInput.length; i++) {
        sum += mfccInput[i];
      }

      float avg = (float)sum / (float)mfccInput.length;

      float total =0;
      for (int i=0; i<mfccInput.length; i++)
        total += (mfccInput[i]-avg)*(mfccInput[i]-avg);

      float dev = total / mfccInput.length; // 분산
      float std = (float)Math.sqrt(dev);

      float eps = (float)1e-6;
      for (int i=0; i<mfccInput.length; i++) {
        //(spec - mean) / (std + eps)
        mfccInput[i] = (mfccInput[i] - avg)/ (std+eps);
      }
      Log.v(LOG_TAG, "MFCC Reged Input======> " + mfccInput[0]);

    // 직접 reshape 해줘야 함
      int SECOND_DIM = 50;
      int THIRD_DIM = 431;
      int FOURTH_DIM = 3; // 그안에서 세번 반
    float[][][][] reshaped_mfccInput = new float[1][50][431][3];
    for(int second_d =0;second_d<SECOND_DIM;second_d++){
      for(int third_d =0;third_d<THIRD_DIM;third_d++){
        // 세번 반복할 애 차원
        float current_input = mfccInput[second_d*SECOND_DIM+third_d];
        for(int fourth_d =0; fourth_d<FOURTH_DIM; fourth_d++){
          reshaped_mfccInput[0][second_d][third_d][fourth_d] = current_input;
        }
      }
    }
//    Log.v(LOG_TAG, "reshaped MFCC Input======> " + reshaped_mfccInput[0].lenth);
//      Log.v(LOG_TAG, "reshaped MFCC Input======> " + reshaped_mfccInput[0][0].lenth);
      Log.v(LOG_TAG, "reshaped MFCC Input======> " + reshaped_mfccInput[0][0][0][0]);
      Log.v(LOG_TAG, "reshaped MFCC Input======> " + reshaped_mfccInput[0][0][0][1]);
      Log.v(LOG_TAG, "reshaped MFCC Input======> " + reshaped_mfccInput[0][0][0][2]);

//      Object[] inputArray = {floatInputBuffer, sampleRateList};
      Object[] inputArray = {reshaped_mfccInput};
      Map<Integer, Object> outputMap = new HashMap<>();
      outputMap.put(0, outputScores);

      // Run the model.
      // inputArray
      tfLite.runForMultipleInputsOutputs(inputArray, outputMap);

      // Use the smoother to figure out if we've had a real recognition event.
      long currentTime = System.currentTimeMillis();
      final RecognizeCommands.RecognitionResult result =
          recognizeCommands.processLatestResults(outputScores[0], currentTime);
      lastProcessingTimeMs = new Date().getTime() - startTime;

      runOnUiThread(
          new Runnable() {
            @Override
            public void run() {

              inferenceTimeTextView.setText(lastProcessingTimeMs + " ms");

              // If we do have a new command, highlight the right list entry.
              if (!result.foundCommand.startsWith("_") && result.isNewCommand) {
                int labelIndex = -1;
                for (int i = 0; i < labels.size(); ++i) {
                  if (labels.get(i).equals(result.foundCommand)) {
                    labelIndex = i;
                  }
                }
                Log.v(LOG_TAG, "label num======> " + labelIndex);
                switch (labelIndex) {
                  case 0:
                    selectedTextView = yesTextView;
                    break;
                  case 1:
                    selectedTextView = noTextView;
                    break;
                  case 2:
                    selectedTextView = upTextView;
                    break;
                  case 3:
                    selectedTextView = downTextView;
                    break;
                  case 4:
                    selectedTextView = leftTextView;
                    break;
                  case 5:
                    selectedTextView = rightTextView;
                    break;
                  case 6:
                    selectedTextView = onTextView;
                    break;
                  case 7:
                    selectedTextView = offTextView;
                    break;
                  case 8:
                    selectedTextView = stopTextView;
                    break;
                  case 9:
                    selectedTextView = goTextView;
                    break;
                }

                if (selectedTextView != null) {
                  selectedTextView.setBackgroundResource(R.drawable.round_corner_text_bg_selected);
                  final String score = Math.round(result.score * 100) + "%";
                  selectedTextView.setText(selectedTextView.getText() + "\n" + score);
                  selectedTextView.setTextColor(
                      getResources().getColor(android.R.color.holo_orange_light));
                  handler.postDelayed(
                      new Runnable() {
                        @Override
                        public void run() {
                          String origionalString =
                              selectedTextView.getText().toString().replace(score, "").trim();
                          selectedTextView.setText(origionalString);
                          selectedTextView.setBackgroundResource(
                              R.drawable.round_corner_text_bg_unselected);
                          selectedTextView.setTextColor(
                              getResources().getColor(android.R.color.darker_gray));
                        }
                      },
                      750);
                }
              }
            }
          });
      try {
        // We don't need to run too frequently, so snooze for a bit.
        Thread.sleep(MINIMUM_TIME_BETWEEN_SAMPLES_MS);
      } catch (InterruptedException e) {
        // Ignore
      }
//    }

    Log.v(LOG_TAG, "End recognition");
  }

  @Override
  public void onClick(View v) {
    if (v.getId() == R.id.plus) {
      String threads = threadsTextView.getText().toString().trim();
      int numThreads = Integer.parseInt(threads);
      numThreads++;
      threadsTextView.setText(String.valueOf(numThreads));
      //            tfLite.setNumThreads(numThreads);
      int finalNumThreads = numThreads;
      backgroundHandler.post(() -> tfLite.setNumThreads(finalNumThreads));
    } else if (v.getId() == R.id.minus) {
      String threads = threadsTextView.getText().toString().trim();
      int numThreads = Integer.parseInt(threads);
      if (numThreads == 1) {
        return;
      }
      numThreads--;
      threadsTextView.setText(String.valueOf(numThreads));
      tfLite.setNumThreads(numThreads);
      int finalNumThreads = numThreads;
      backgroundHandler.post(() -> tfLite.setNumThreads(finalNumThreads));
    }
  }

  @Override
  public void onCheckedChanged(CompoundButton buttonView, boolean isChecked) {
    backgroundHandler.post(() -> tfLite.setUseNNAPI(isChecked));
    if (isChecked) apiSwitchCompat.setText("NNAPI");
    else apiSwitchCompat.setText("TFLITE");
  }

  private static final String HANDLE_THREAD_NAME = "CameraBackground";

  private void startBackgroundThread() {
    backgroundThread = new HandlerThread(HANDLE_THREAD_NAME);
    backgroundThread.start();
    backgroundHandler = new Handler(backgroundThread.getLooper());
  }

  private void stopBackgroundThread() {
    backgroundThread.quitSafely();
    try {
      backgroundThread.join();
      backgroundThread = null;
      backgroundHandler = null;
    } catch (InterruptedException e) {
      Log.e("amlan", "Interrupted when stopping background thread", e);
    }
  }

  @Override
  protected void onResume() {
    super.onResume();

    startBackgroundThread();
  }

  @Override
  protected void onStop() {
    super.onStop();
    stopBackgroundThread();
  }
}
