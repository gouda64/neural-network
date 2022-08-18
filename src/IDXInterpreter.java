import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.*;

public class IDXInterpreter {
    public static int[][] imageRead(String imageFile) {
        try {
            FileInputStream imageContent = new FileInputStream(imageFile);

            int magicNum = (imageContent.read() << 24) | (imageContent.read() << 16) | (imageContent.read() << 8) | imageContent.read();
            //we're just assuming the information is in unsigned bits and 2-dimensional here so magicNum isn't really needed
            int numItems = (imageContent.read() << 24) | (imageContent.read() << 16) | (imageContent.read() << 8) | imageContent.read();
            int numRows = (imageContent.read() << 24) | (imageContent.read() << 16) | (imageContent.read() << 8) | imageContent.read();
            int numCols = (imageContent.read() << 24) | (imageContent.read() << 16) | (imageContent.read() << 8) | imageContent.read();

            int[][] imgs = new int[numItems][numRows * numCols];
            for (int i = 0; i < numItems; i++) {
                for (int j = 0; j < numRows * numCols; j++) {
                    imgs[i][j] = imageContent.read();
                }
            }
            imageContent.close();
            return imgs;
        }
        catch(IOException e) {
            e.printStackTrace();
            return null;
        }
    }
    public static BufferedImage imgFromArray(int[] pixels, int width, int height) {
        BufferedImage bi = new BufferedImage(width, height, BufferedImage.TYPE_BYTE_GRAY);
        Graphics2D g = (Graphics2D) bi.getGraphics();
        for (int i = 0; i < pixels.length; i++) {
            int gray = pixels[i];
            g.setColor(new Color(gray, gray, gray));
            g.drawRect(i%width, i/height, 1,1);
        }
        return bi;
    }

    public static int[] labelRead(String labelFile) {
        try {
            FileInputStream labelContent = new FileInputStream(labelFile);

            int magicNum = (labelContent.read() << 24) | (labelContent.read() << 16) | (labelContent.read() << 8) | labelContent.read();
            //we're just assuming the information is in unsigned bits and 1-dimensional here so magicNum isn't really needed
            int numItems = (labelContent.read() << 24) | (labelContent.read() << 16) | (labelContent.read() << 8) | labelContent.read();

            int[] labels = new int[numItems];
            for (int i = 0; i < numItems; i++) {
                labels[i] = labelContent.read();
            }

            labelContent.close();
            return labels;
        }
        catch (IOException e) {
            e.printStackTrace();
            return null;
        }
    }
}
