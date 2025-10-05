# Use more consistent and modern versions
ARG IMAGE_VARIANT=slim-bullseye
ARG OPENJDK_VERSION=11
ARG SPARK_VERSION=3.5.1
ARG HADOOP_VERSION=3

FROM openjdk:${OPENJDK_VERSION}-${IMAGE_VARIANT}

ARG SPARK_VERSION
ARG HADOOP_VERSION

# Install Python
RUN apt-get update && apt-get install -y python3 python3-pip curl procps && \
    rm -rf /var/lib/apt/lists/*

# Install PySpark
RUN pip3 --no-cache-dir install pyspark==${SPARK_VERSION}

# Download and install the full Spark binaries
ENV SPARK_HOME=/opt/spark
ENV PATH=$SPARK_HOME/bin:$PATH

RUN curl -fSL "https://archive.apache.org/dist/spark/spark-${SPARK_VERSION}/spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION}.tgz" -o /tmp/spark.tgz && \
    tar -xvf /tmp/spark.tgz -C /opt/ && \
    mv /opt/spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION} ${SPARK_HOME} && \
    rm /tmp/spark.tgz

WORKDIR /app

COPY . .

CMD ["/bin/bash"]