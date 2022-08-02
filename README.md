# Setup guide

1. Use the makefile to build the docker container and setup the project:

```
make build-containers
```

Then use some tool (I use VSCode) to enter the container and work from there.

2. Use the Makefile to create splits from the raw data.
```
make build-datasets
```

4. Log in to W&B
```
make setup-wandb
```
5.  Train model cv
```
make train-cv
```

6. Evaluate model in test (last 80 days)
```
make score
```