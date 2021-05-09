$(document).ready(() => {

    const realBtn = document.getElementById('real-file');
    const customBtn = document.getElementById('custom-btn');
    const customTxt = document.getElementById('custom-span');
    const customSize = document.getElementById('custom-size');
    const uploadSubmit = document.getElementById('submit');
    const uploadBtn = document.getElementById('upload-btn');

    // Change the action when clicking the input button to the custom button
    customBtn.addEventListener('click', function(){
        realBtn.click();
    });

    uploadBtn.addEventListener('click', function(){
        uploadSubmit.click();
    });

    // Function for getting image size
    function getFileSize(file){
        var file = file.files[0];
        if(file.size > (1024 * 1024)) return Math.round(file.size / (1024 * 1024)) + ' MB';
        if(file.size > 1024) return Math.round(file.size / 1024) + ' KB';
        return file.size + ' B';
    };

    realBtn.addEventListener('change', function(){
        var reader = new FileReader();

        reader.onload = function(){
            if($('.transfer').attr('src') == ''){
                $('.transfer').attr('src', reader.result).show();
            }


            // var img = new Image();
            // img.src = reader.result;
            // document.body.appendChild(img);
        }

        reader.readAsDataURL(realBtn.files[0]);
    });

    realBtn.addEventListener('change', function() {
        if(realBtn.value){
            var txt = realBtn.value.split('\\').pop().split('/').pop();
            customTxt.innerHTML = txt;
            customSize.innerHTML = getFileSize(realBtn);
        }
        else{
            customTxt.innerHTML = 'No file chosen';
        }
    });
});


// Return filename + extension from a input type = file
// realBtn.value.split('\\').pop().split('/').pop()
// Return filename + extension from a given path
// console.log(path.split('/').pop())
