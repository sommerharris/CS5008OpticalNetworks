package test;


import org.junit.Test;
import org.junit.Assert.*;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;

import static org.junit.Assert.assertEquals;

public class Qtest {
    @Test
    public void testArgMax(){
        INDArray array = Nd4j.zeros(2, 3, 4, 5);
        array.putScalar(new int[]{0, 1, 2, 0}, 9);
        array.putScalar(new int[]{0, 1, 2, 1}, 1000);
        array.putScalar(new int[]{0, 1, 2, 2}, 80);
        array.putScalar(new int[]{0, 1, 2, 3}, 700000);
        array.putScalar(new int[]{0, 1, 2, 4}, 60);
        INDArray array1 = array.get(NDArrayIndex.point(0),
                NDArrayIndex.point(1),
                NDArrayIndex.point(2));
        INDArray indArray = array1.argMax();
        System.out.println(indArray.getInt(0));
        int[] values = array1.toIntVector();

        for (int i=0; i<5; i++) {
            assertEquals(values[i], array1.getInt(i));
        }

    }

}
