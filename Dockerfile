# Alpine has the most reliable mirrors and smallest size
ARG SPARK_VERSION=3.5.1
ARG HADOOP_VERSION=3

FROM alpine:3.19

ARG SPARK_VERSION
ARG HADOOP_VERSION

# Install Java 11, Python, and dependencies
RUN apk add --no-cache \
    openjdk11-jre \
    python3 \
    py3-pip \
    py3-numpy \
    py3-pandas \
    bash \
    curl \
    procps \
    shadow && \
    ln -sf python3 /usr/bin/python

# Install PySpark
RUN pip3 --no-cache-dir install --break-system-packages pyspark==${SPARK_VERSION}

# Download and install the full Spark binaries
ENV SPARK_HOME=/opt/spark
ENV PATH=$SPARK_HOME/bin:$PATH
ENV JAVA_HOME=/usr/lib/jvm/java-11-openjdk

RUN curl -fSL "https://archive.apache.org/dist/spark/spark-${SPARK_VERSION}/spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION}.tgz" -o /tmp/spark.tgz && \
    mkdir -p ${SPARK_HOME} && \
    tar -xzf /tmp/spark.tgz -C /opt/ && \
    mv /opt/spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION}/* ${SPARK_HOME}/ && \
    rm -rf /tmp/spark.tgz /opt/spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION}

WORKDIR /app

COPY . .

CMD ["/bin/bash"]