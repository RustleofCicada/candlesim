function render() {
    var img = document.getElementById('bitmap');
    fetch('/stream')
    .then(response => response.blob())
    .then(images =>
        {
            var outside = URL.createObjectURL(images);
            img.src = outside;
        });
    setTimeout(render, 200);
}
