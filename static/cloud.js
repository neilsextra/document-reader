function Cloud(account, token, container, directory) {

    this.__account = account;
    this.__token = token;
    this.__container = container;
    this.__directory = directory;

}

Cloud.prototype.query = function() {

    return new Promise((accept, reject) => {
        var xhttp = new XMLHttpRequest();

        xhttp.open("GET", `/query?account=${encodeURIComponent(this.__account)}&token=${encodeURIComponent(this.__token)}` +
            `&container=${encodeURIComponent(this.__container)}&directory=${encodeURIComponent(this.__directory)}`, true);

        xhttp.onreadystatechange = async function() {

            if (this.readyState === 4 && this.status === 200) {
                var paths = [];
                var response = JSON.parse(this.responseText);

                accept({
                    status: this.status,
                    metadata: response[0]['directory_metadata'],
                    paths: response[0]['paths']
                });

            } else if (this.status === 500) {

                reject({
                    status: this.status,
                    message: this.statusText
                });

            }

        };

        xhttp.send();

    });

}

Cloud.prototype.setup = function(formData) {

    formData.append("cloud_account", this.__account);
    formData.append("cloud_token", this.__token);
    formData.append("cloud_container", this.__container);
    formData.append("cloud_directory", this.__directory);

}

Cloud.prototype.retreive = function(filename) {

    return new Promise((accept, reject) => {

        fetch(`/retrieve?account=${encodeURIComponent(this.__account)}&token=${encodeURIComponent(this.__token)}` +
                `&container=${encodeURIComponent(this.__container)}` +
                `&directory=${encodeURIComponent(this.__directory)}&filename=${encodeURIComponent(filename)}`, {
                    responseType: 'blob'
                })
            .then(res => res.blob())
            .then(blob => {
                accept(blob.arrayBuffer())
            });

    })

}

Cloud.prototype.train = function(trainingURL, apimKey) {

    return new Promise((accept, reject) => {

        fetch(`/train?account=${encodeURIComponent(this.__account)}&token=${encodeURIComponent(this.__token)}` +
                `&container=${encodeURIComponent(this.__container)}` +
                `&directory=${encodeURIComponent(this.__directory)}` +
                `&formURL=${encodeURIComponent(trainingURL)}` +
                `&apimKey=${encodeURIComponent(apimKey)}`
            )
            .then(res => res.text())
            .then(text => {
                accept(text)
            });

    })

}

Cloud.prototype.analyze = function(filename) {

    return new Promise((accept, reject) => {

        fetch(`/analyze?account=${encodeURIComponent(this.__account)}&token=${encodeURIComponent(this.__token)}` +
                `&container=${encodeURIComponent(this.__container)}` +
                `&directory=${encodeURIComponent(this.__directory)}` +
                `&filename=${encodeURIComponent(filename)}`
            )
            .then(res => res.text())
            .then(text => {
                accept(text)
            });

    })

}