"""Turn hand-marked transitions (tools/labeler.html export) into a full frame,label ground truth.

    python tools/transitions_to_gt.py hand_labels_transitions.csv cache/hand_gt.csv
    python external/more-than-a-hack-2027/evaluation/run_evaluation.py --gt cache/hand_gt.csv \
        --results submissions

Each mark starts a state at its frame; X marks end a labelled stretch (unlabelled = X = never
scored). Frames outside the graded halves are forced to X. Keep the output in cache/ (gitignored):
hand labels of the match are frame-level derived data.
"""
import sys

N = 199319
HALVES = [(11185, 90651), (110808, 190197)]


def main(src, dst):
    marks = []
    with open(src) as fh:
        next(fh)
        for line in fh:
            parts = line.strip().split(",")
            if len(parts) == 2 and parts[1].strip().upper() in "HADX":
                marks.append((int(parts[0]), parts[1].strip().upper()))
    marks.sort()
    labels = ["X"] * N
    for i, (frame, lab) in enumerate(marks):
        end = marks[i + 1][0] if i + 1 < len(marks) else N
        for f in range(frame, min(end, N)):
            labels[f] = lab
    graded = [False] * N
    for a, b in HALVES:
        for f in range(a, b + 1):
            graded[f] = True
    labels = [l if graded[f] else "X" for f, l in enumerate(labels)]
    with open(dst, "w", newline="\n") as fh:
        fh.write("frame,label\n")
        fh.writelines("{},{}\n".format(f, l) for f, l in enumerate(labels))
    n_lab = sum(l != "X" for l in labels)
    print("wrote {}: {} labelled frames ({:.1f} min)".format(dst, n_lab, n_lab / 25 / 60))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
