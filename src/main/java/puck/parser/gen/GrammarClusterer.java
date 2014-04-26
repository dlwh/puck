package puck.parser.gen;

import java.util.List;

/**
 * TODO
 *
 * @author dlwh
 */
public interface GrammarClusterer<C, L> {
    public static final int NUM_SM = 8;

    public abstract List<IndexedUnaryRule<C, L>>[] segmentUnaries(List<IndexedUnaryRule<C, L>> indexedUnaryRules);

    public abstract List<IndexedBinaryRule<C, L>>[][] segmentBinaries(List<IndexedBinaryRule<C, L>> indexedBinaryRules);
}
