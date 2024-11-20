export default function(route) {
    let url = window.reverseUrl(route);

    // Use https in production
    if (process.env.NODE_ENV === 'production') {
        url = url.replace('http://', 'https://');
    }

    return url;
}