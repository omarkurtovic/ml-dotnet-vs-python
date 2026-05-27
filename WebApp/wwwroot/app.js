window.setCultureCookie = function (culture) {
    document.cookie = `.AspNetCore.Culture=c=${culture}|uic=${culture}; path=/; max-age=31536000`;
};