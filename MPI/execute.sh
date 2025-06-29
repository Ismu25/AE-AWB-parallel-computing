mpirun --bind-to none -np $1 -host localhost:$1 MPI ${@:2}
