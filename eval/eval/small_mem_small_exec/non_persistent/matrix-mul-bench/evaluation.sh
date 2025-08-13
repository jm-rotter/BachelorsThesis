echo "Computing average over 10000 runs"

rm -f results.txt

for i in {1..10000}
do
  ./matrix_multiplication | grep -oE '[0-9]+\.[0-9]+' >> results.txt
done

awk '{ sum += $1; count++ } END { if (count > 0) print sum/count; else print "No data" }' results.txt
