// document.querySelector('.down-arrow').addEventListener('click', function () {
//     // Find the first H1 element
//     const nextHeading = document.querySelector('h1');

//     if (nextHeading) {
//         // Calculate scroll position to align the top of the H1 with the top of the viewport
//         // Subtract the height of any fixed headers or desired offset
//         const headerOffset =  document.querySelector('header').offsetHeight; // Adjust this value if you have a fixed header or want some padding
//         const scrollPosition = nextHeading.getBoundingClientRect().top + window.scrollY - headerOffset;

//         window.scrollTo({
//             top: scrollPosition,
//             behavior: 'smooth'
//         });
//     }
// });
document.querySelector('.down-arrow').addEventListener('click', function () {
    // Scroll exactly one full viewport height
    window.scrollTo({
        top: window.innerHeight,
        behavior: 'smooth'
    });
});
