for day in day01 day02 day03 day04 day06
do
    python_out=$(./runpy.sh $day)
    cuda_out=$(./runcu.sh $day)
    if [ "$python_out" = "$cuda_out" ];
    then
        echo $day pass!
    else
        echo $day FAIL!
        echo "$python_out"
        echo "$cuda_out"
    fi
done
